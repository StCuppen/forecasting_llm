CREATE TABLE IF NOT EXISTS questions (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  close_time TEXT,
  resolve_time_expected TEXT,
  tags_json TEXT NOT NULL,
  resolver_type TEXT NOT NULL,
  resolver_config_json TEXT NOT NULL,
  status TEXT NOT NULL,
  duplicate_of TEXT,
  raw_payload_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(source, source_id)
);

CREATE TABLE IF NOT EXISTS question_tags (
  question_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  PRIMARY KEY (question_id, tag),
  FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  made_at TEXT NOT NULL,
  p_ens REAL NOT NULL,
  p_agents_json TEXT NOT NULL,
  model_versions_json TEXT NOT NULL,
  evidence_bundle_id TEXT NOT NULL,
  cost_estimate REAL,
  latency REAL,
  forecast_context_json TEXT NOT NULL,
  calibrator_version TEXT,
  UNIQUE(question_id, run_id),
  FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_bundles (
  bundle_id TEXT PRIMARY KEY,
  items_json TEXT NOT NULL,
  archived_text_hashes_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resolutions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question_id TEXT NOT NULL UNIQUE,
  resolved_at TEXT NOT NULL,
  y REAL NOT NULL,
  resolver_payload_raw_json TEXT NOT NULL,
  resolution_confidence REAL NOT NULL,
  status TEXT NOT NULL,
  FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question_id TEXT NOT NULL UNIQUE,
  scored_at TEXT NOT NULL,
  brier_ens REAL NOT NULL,
  logloss_ens REAL NOT NULL,
  brier_agents_json TEXT NOT NULL,
  logloss_agents_json TEXT NOT NULL,
  aggregates_json TEXT NOT NULL,
  FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS diagnostics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  question_id TEXT NOT NULL UNIQUE,
  error_type TEXT NOT NULL,
  structured_notes_json TEXT NOT NULL,
  recommended_patch TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(question_id) REFERENCES questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS weights (
  domain_tag TEXT NOT NULL,
  agent_name TEXT NOT NULL,
  weight REAL NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(domain_tag, agent_name)
);

CREATE TABLE IF NOT EXISTS calibrators (
  domain_tag TEXT PRIMARY KEY,
  version TEXT NOT NULL,
  fitted_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS updater_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status);
CREATE INDEX IF NOT EXISTS idx_questions_resolve_time_expected ON questions(resolve_time_expected);
CREATE INDEX IF NOT EXISTS idx_questions_source ON questions(source);
CREATE INDEX IF NOT EXISTS idx_question_tags_tag ON question_tags(tag);
CREATE INDEX IF NOT EXISTS idx_predictions_made_at ON predictions(made_at);
