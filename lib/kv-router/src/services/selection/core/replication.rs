// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-carried selection-service updates. Worker lifecycle stays with the host.

use serde::{Deserialize, Serialize};

use super::*;
use crate::services::selection::PromptRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "event", rename_all = "snake_case")]
pub enum ReplicaUpdate {
    Reservation(ReservationReplicaEvent),
    Affinity(AffinityBindingEvent),
}

impl From<AffinityBindingEvent> for ReplicaUpdate {
    fn from(event: AffinityBindingEvent) -> Self {
        Self::Affinity(event)
    }
}

/// A finalized booking; importing it neither binds a session nor publishes an echo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservationReplicaEvent {
    #[serde(flatten)]
    pub partition: RoutingPartitionId,
    pub writer_id: u64,
    pub block_size: u32,
    pub selection_id: String,
    pub worker_id: WorkerId,
    pub dp_rank: u32,
    pub sequence_hashes: Vec<i64>,
    pub isl_tokens: usize,
    pub expected_output_tokens: Option<u32>,
    pub effective_prefill_tokens: Option<usize>,
    pub track_prefill_tokens: bool,
    pub lora_name: Option<String>,
}

impl SelectionCore {
    pub(super) fn publish_reservation(&self, event: ReservationReplicaEvent) {
        if let Some(tx) = &self.host.replication.outbound
            && let Err(error) = tx.try_send(ReplicaUpdate::Reservation(event))
        {
            tracing::trace!(%error, "dropping best-effort reservation replica update");
        }
    }

    pub(crate) async fn apply_replica_update(
        &self,
        update: ReplicaUpdate,
    ) -> Result<ReplicaApplyOutcome, SelectionError> {
        let event = match update {
            ReplicaUpdate::Affinity(event) => return Ok(self.dispatch_affinity_event(event)),
            ReplicaUpdate::Reservation(event) => event,
        };
        if self.host.replication.outbound.is_none() {
            return Err(SelectionError::Conflict(
                "host replica stream is disabled".into(),
            ));
        }
        if event.writer_id == self.host.replication.writer_id {
            return Ok(ReplicaApplyOutcome::IgnoredSelf);
        }
        let Some(entry) = self.entry(&event.partition) else {
            return Ok(ReplicaApplyOutcome::UnknownPartition);
        };
        if event.block_size != entry.block_size {
            return Err(SelectionError::BadRequest(
                "replica block size mismatch".into(),
            ));
        }
        if !self.catalog.is_schedulable(
            WorkerWithDpRank::new(event.worker_id, event.dp_rank).into(),
            &event.partition,
        ) {
            return Ok(ReplicaApplyOutcome::UnknownWorker);
        }
        let request = ReservationRequest {
            model_name: event.partition.model_name.clone(),
            routing_group: event.partition.routing_group.clone(),
            selection_id: event.selection_id,
            worker_id: Some(event.worker_id),
            dp_rank: Some(event.dp_rank),
            prompt: PromptRequest {
                sequence_hashes: Some(event.sequence_hashes),
                isl_tokens: Some(event.isl_tokens),
                lora_name: event.lora_name,
                ..PromptRequest::default()
            },
            router_config_override: None,
            expected_output_tokens: event.expected_output_tokens,
            effective_prefill_tokens: event.effective_prefill_tokens,
            track_prefill_tokens: Some(event.track_prefill_tokens),
        };
        match self
            .reserve_explicit(event.partition, event.worker_id, request, false)
            .await
        {
            Ok(_) => Ok(ReplicaApplyOutcome::Applied),
            Err(SelectionError::Conflict(_)) => Ok(ReplicaApplyOutcome::IgnoredConflict),
            Err(error) => Err(error),
        }
    }
}
