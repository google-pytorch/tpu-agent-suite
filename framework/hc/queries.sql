-- Copyright 2026 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Named SQL queries for Hillclimber coordination operations.
-- Every parameter uses SQLite `:name` syntax; positional `?` bindings are forbidden.

-- name: has_user_tables
SELECT 1
FROM sqlite_master
WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
LIMIT 1;

-- name: list_dispatched_jobs
SELECT j.job_id, j.kind, j.round
FROM Jobs j
JOIN Candidates c ON c.id = j.candidate
WHERE c.campaign_id = :campaign_id AND j.status = :status
ORDER BY j.rowid ASC;

-- name: update_job_status
UPDATE Jobs
SET status = :status
WHERE job_id = :job_id;

-- name: insert_protocol
INSERT INTO Protocols (
    workload_kind, primary_metric, compute_profile, primary_metric_unit,
    primary_metric_direction, primary_metric_statistic, min_replicates,
    min_win_percentage, max_cell_regression_percentage, noise_floor_percentage,
    max_rounds, max_barren_rounds, target_improvement_percent,
    has_accuracy_gate, accuracy_gate_metric, accuracy_gate_comparator,
    accuracy_gate_threshold, accuracy_gate_is_relative, accuracy_gate_relative_to,
    exec_image, exec_microbenchmark_target, exec_base_server_args,
    exec_base_client_args, exec_base_eval_args
) VALUES (
    :workload_kind, :primary_metric, :compute_profile, :primary_metric_unit,
    :primary_metric_direction, :primary_metric_statistic, :min_replicates,
    :min_win_percentage, :max_cell_regression_percentage, :noise_floor_percentage,
    :max_rounds, :max_barren_rounds, :target_improvement_percent,
    :has_accuracy_gate, :accuracy_gate_metric, :accuracy_gate_comparator,
    :accuracy_gate_threshold, :accuracy_gate_is_relative, :accuracy_gate_relative_to,
    :exec_image, :exec_microbenchmark_target, :exec_base_server_args,
    :exec_base_client_args, :exec_base_eval_args
);

-- name: insert_protocol_cell
INSERT INTO Protocol_Cells (
    protocol_id, cell_index, topology, weight, prefill, decode,
    concurrency, sequence_length, global_batch_size, per_device_batch_size
) VALUES (
    :protocol_id, :cell_index, :topology, :weight, :prefill, :decode,
    :concurrency, :sequence_length, :global_batch_size, :per_device_batch_size
);

-- name: update_protocol_microbenchmark_target
UPDATE Protocols
SET exec_microbenchmark_target = :target
WHERE id = :id;

-- name: get_campaign_by_id
SELECT *
FROM Campaigns
WHERE id = :id;

-- name: insert_campaign
INSERT INTO Campaigns (
    name, owner, target_model, engine, accelerator, created_at, closed_at,
    protocol, microbenchmark_protocol, champion, initial_version,
    entry_baseline, current_baseline, close_reason,
    num_consecutive_barren_rounds, state, block_reason
) VALUES (
    :name, :owner, :target_model, :engine, :accelerator, :created_at, NULL,
    :protocol, :microbenchmark_protocol, NULL, 0, NULL, NULL, NULL, 0,
    :state, NULL
);

-- name: set_campaign_initial_version
UPDATE Campaigns
SET initial_version = :initial_version
WHERE id = :id;

-- name: count_campaign_jobs_by_kind
SELECT COUNT(*) AS cnt
FROM Jobs j
JOIN Candidates c ON c.id = j.candidate
WHERE c.campaign_id = :campaign_id AND j.kind = :kind;

-- name: set_campaign_microbenchmark_protocol
UPDATE Campaigns
SET microbenchmark_protocol = :protocol_id
WHERE id = :id;

-- name: update_campaign_block_state
UPDATE Campaigns
SET state = :state, block_reason = :block_reason
WHERE id = :id;

-- name: close_campaign
UPDATE Campaigns
SET state = :state, close_reason = :close_reason, closed_at = :closed_at, block_reason = NULL
WHERE id = :id;

-- name: get_open_round
SELECT *
FROM Rounds
WHERE campaign = :campaign_id AND state = :state;

-- name: get_max_round_number
SELECT COALESCE(MAX(round_number), 0) AS max_num
FROM Rounds
WHERE campaign = :campaign_id;

-- name: insert_round
INSERT INTO Rounds (campaign, round_number, current_champion, state)
VALUES (:campaign_id, :round_number, :current_champion, :state);

-- name: get_campaign_hypothesis
SELECT h.*
FROM Hypotheses h
JOIN Rounds r ON r.id = h.round_created
WHERE h.id = :hypothesis_id AND r.campaign = :campaign_id;

-- name: insert_hypothesis
INSERT INTO Hypotheses (
    round_created, lever_kind, target_scope, short_claim, state, claim, refutation_reason
) VALUES (
    :round_created, :lever_kind, :target_scope, :short_claim, :state, :claim, NULL
);

-- name: insert_hypothesis_canonical_op
INSERT INTO Hypothesis_Canonical_Ops (hypothesis_id, op_index, layer_kind, operator)
VALUES (:hypothesis_id, :op_index, :layer_kind, :operator);

-- name: insert_hypothesis_relayout_edge
INSERT INTO Hypothesis_Relayout_Edges (
    hypothesis_id, edge_index, producer_layer_kind, producer_operator,
    consumer_layer_kind, consumer_operator
) VALUES (
    :hypothesis_id, :edge_index, :producer_layer_kind, :producer_operator,
    :consumer_layer_kind, :consumer_operator
);

-- name: insert_hypothesis_evidence
INSERT INTO Hypothesis_Evidence (
    hypothesis_id, evidence_index, source_type, locator, summary
) VALUES (
    :hypothesis_id, :evidence_index, :source_type, :locator, :summary
);

-- name: insert_hypothesis_code_ref
INSERT INTO Hypothesis_Code_Refs (hypothesis_id, ref_index, code_ref)
VALUES (:hypothesis_id, :ref_index, :code_ref);

-- name: refute_hypothesis
UPDATE Hypotheses
SET state = :state, refutation_reason = :reason
WHERE id = :id;

-- name: get_campaign_candidate
SELECT *
FROM Candidates
WHERE id = :id AND campaign_id = :campaign_id;

-- name: insert_candidate
INSERT INTO Candidates (
    campaign_id, round, sha, tree_sha, patch_sha,
    short_innovation, superseded_by, innovation, abandoned_reason
) VALUES (
    :campaign_id, :round, :sha, :tree_sha, :patch_sha,
    :short_innovation, NULL, :innovation, NULL
);

-- name: insert_candidate_parent
INSERT INTO Candidate_Parents (candidate_id, parent_index, parent_id)
VALUES (:candidate_id, :parent_index, :parent_id);

-- name: insert_hypothesis_candidate
INSERT INTO Hypothesis_Candidates (hypothesis_id, candidate_id)
VALUES (:hypothesis_id, :candidate_id);

-- name: set_candidate_superseded_by
UPDATE Candidates
SET superseded_by = :superseded_by
WHERE id = :id;

-- name: abandon_candidate
UPDATE Candidates
SET abandoned_reason = :reason
WHERE id = :id;

-- name: insert_review
INSERT INTO Reviews (
    candidate, hypothesis, human_authored, reviewer, approval,
    short_comments, conditions, comments
) VALUES (
    :candidate, :hypothesis, :human_authored, :reviewer, :approval,
    :short_comments, :conditions, :comments
);
