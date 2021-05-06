#pragma once

// constexpr int kNumColors = 32;

constexpr int kMaxCoastCycles = 3;  // maximum time to continue tracking without fitting observation

constexpr int kMinHits = 1;   // minimum times updated by an observation that fit the prediction (to ignore false positives)

// // Set threshold to 0 to accept all detections
// constexpr float kMinConfidence = 0.6;
