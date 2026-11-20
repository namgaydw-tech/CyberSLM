export type EventRecord = Record<string, unknown>;

export type EvidenceSummary = {
  rowsAvailable: number;
  eventTypeCounts: Record<string, number>;
  distinctSources: number | null;
  distinctDestinations: number | null;
  verification: string;
};

export type ApiResult = {
  ok: boolean;
  mode: string;
  content: string;
  provider?: string;
  model?: string;
  evidence?: EvidenceSummary;
  scratchPrediction?: string;
  verifiedAnswer?: string | null;
  verificationStatus?: string;
  attempts?: string[];
};
