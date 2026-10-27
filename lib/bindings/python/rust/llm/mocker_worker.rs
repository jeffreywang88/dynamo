// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A single live Mocker worker that runs without a `DistributedRuntime`.
//!
//! External hosts (for example a Ray Serve replica) own discovery, transport and
//! request routing themselves. They only need the simulated scheduler, its KV
//! events on a vLLM-compatible ZMQ publisher, and per-request output. Output is
//! coalesced in Rust so the host pays per flush, not per token.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use dynamo_kv_router::protocols::KvCacheEventData;
use dynamo_mocker::common::protocols::{
    DirectRequest, FpmPublisher, KvEventPublishers, RawKvEvent, RawKvEventSink,
};
use dynamo_mocker::live::{LiveEngine, LiveEngineConfig, LiveRequest, RequestOutputBuffering};
use dynamo_mocker::services::zmq_events::ZmqKvEventSink;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::sync::Mutex;

use super::replay::{materialize_mocker_config, mocker_config_from_python};

/// Counts the KV blocks a worker stores and removes, then forwards to ZMQ.
struct CountingSink {
    inner: ZmqKvEventSink,
    stored_blocks: AtomicU64,
    removed_blocks: AtomicU64,
}

impl CountingSink {
    fn count(&self, event: &RawKvEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(data) => {
                self.stored_blocks
                    .fetch_add(data.blocks.len() as u64, Ordering::Relaxed);
            }
            KvCacheEventData::Removed(data) => {
                self.removed_blocks
                    .fetch_add(data.block_hashes.len() as u64, Ordering::Relaxed);
            }
            KvCacheEventData::Cleared => {}
        }
    }
}

impl RawKvEventSink for CountingSink {
    fn publish(&self, event: RawKvEvent) -> anyhow::Result<()> {
        self.count(&event);
        self.inner.publish(event)
    }

    fn publish_batch(&self, events: Vec<RawKvEvent>) -> anyhow::Result<()> {
        for event in &events {
            self.count(event);
        }
        self.inner.publish_batch(events)
    }
}

/// One live Mocker scheduler publishing KV events on `tcp://*:{zmq_port}`.
#[pyclass]
pub struct MockerWorker {
    engine: LiveEngine,
    sink: Arc<CountingSink>,
}

#[pymethods]
impl MockerWorker {
    /// `config` is a canonical Mocker config mapping (or its JSON string), as
    /// produced by `dynamo.mocker.config.load_mocker_engine_args`.
    #[new]
    #[pyo3(signature = (config, zmq_port, zmq_replay_port=None))]
    fn new(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        zmq_port: u16,
        zmq_replay_port: Option<u16>,
    ) -> PyResult<Self> {
        let args = materialize_mocker_config(py, mocker_config_from_python(py, config)?)?;
        if args.dp_size != 1 {
            return Err(PyValueError::new_err("MockerWorker supports dp_size=1 only"));
        }
        let block_size = args.block_size as u32;
        let runtime = crate::bridge_runtime();
        py.allow_threads(|| {
            let _guard = runtime.enter();
            let inner = ZmqKvEventSink::bind(Some(zmq_port), zmq_replay_port, 0, block_size)
                .map_err(|e| PyRuntimeError::new_err(format!("ZMQ KV sink: {e:#}")))?;
            let sink = Arc::new(CountingSink {
                inner,
                stored_blocks: AtomicU64::new(0),
                removed_blocks: AtomicU64::new(0),
            });
            let config = LiveEngineConfig {
                kv_event_publishers: KvEventPublishers::new(
                    None,
                    Some(sink.clone() as Arc<dyn RawKvEventSink>),
                ),
                fpm_publisher: FpmPublisher::default(),
            };
            let mut engines = LiveEngine::start_grouped_with_configs_and_request_output_buffering(
                args,
                vec![config],
                RequestOutputBuffering::FullResponse,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("start Mocker: {e:#}")))?;
            let engine = engines
                .pop()
                .ok_or_else(|| PyRuntimeError::new_err("Mocker returned no engine"))?;
            Ok(Self { engine, sink })
        })
    }

    /// Submit a request; resolves to a `MockerRequest` once it is admitted.
    fn submit<'p>(
        &self,
        py: Python<'p>,
        token_ids: Vec<u32>,
        max_tokens: usize,
    ) -> PyResult<Bound<'p, PyAny>> {
        let engine = self.engine.clone();
        crate::future_into_py(py, async move {
            let request = DirectRequest {
                tokens: token_ids,
                max_output_tokens: max_tokens,
                ..Default::default()
            };
            let live = engine
                .submit(request)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
            Ok(MockerRequest {
                live: Arc::new(Mutex::new(live)),
                emitted: Arc::new(AtomicU64::new(0)),
            })
        })
    }

    /// `(stored_blocks, removed_blocks)` published since start.
    fn kv_event_counts(&self) -> (u64, u64) {
        (
            self.sink.stored_blocks.load(Ordering::Relaxed),
            self.sink.removed_blocks.load(Ordering::Relaxed),
        )
    }

    fn active_request_count(&self) -> usize {
        self.engine.active_request_count()
    }
}

/// Output of one submitted request.
#[pyclass]
pub struct MockerRequest {
    live: Arc<Mutex<LiveRequest>>,
    emitted: Arc<AtomicU64>,
}

#[pymethods]
impl MockerRequest {
    /// Wait for the next output chunk.
    ///
    /// Resolves to `(new_tokens, cached_tokens, done)`. The first token is
    /// returned as soon as it is produced; later tokens are batched for up to
    /// `flush_interval_s`. `cached_tokens` is set once, with the first token.
    #[pyo3(signature = (flush_interval_s=0.05))]
    fn next_chunk<'p>(&self, py: Python<'p>, flush_interval_s: f64) -> PyResult<Bound<'p, PyAny>> {
        let live = self.live.clone();
        let emitted = self.emitted.clone();
        crate::future_into_py(py, async move {
            let mut live = live.lock().await;
            let mut new_tokens = 0u64;
            let mut cached_tokens: Option<usize> = None;
            let mut done = false;
            let mut deadline: Option<tokio::time::Instant> = None;
            loop {
                let signal = match deadline {
                    None => live.recv().await,
                    Some(at) => match tokio::time::timeout_at(at, live.recv()).await {
                        Ok(signal) => signal,
                        Err(_) => break,
                    },
                };
                let Some(signal) = signal else {
                    done = true;
                    break;
                };
                if signal.cached_tokens.is_some() {
                    cached_tokens = signal.cached_tokens;
                }
                if signal.token_id.is_some() {
                    new_tokens += 1;
                }
                if signal.completed {
                    done = true;
                    break;
                }
                if emitted.load(Ordering::Relaxed) + new_tokens == 1 {
                    // First token: return it right away so TTFT is exact.
                    break;
                }
                if deadline.is_none() {
                    deadline = Some(
                        tokio::time::Instant::now() + Duration::from_secs_f64(flush_interval_s),
                    );
                }
            }
            emitted.fetch_add(new_tokens, Ordering::Relaxed);
            Ok((new_tokens, cached_tokens, done))
        })
    }
}
