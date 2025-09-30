package ann

// Re-export shared metrics for backward compatibility
import (
	"github.com/pilillo/yannp/shared/metrics"
)

// Metric interface re-exported from shared package
type Metric = metrics.Metric

// NewAccuracyMetric re-exported from shared package
var NewAccuracyMetric = metrics.NewAccuracyMetric

// ClassificationMetrics re-exported from shared package
type ClassificationMetrics = metrics.ClassificationMetrics

// NewClassificationMetrics re-exported from shared package
var NewClassificationMetrics = metrics.NewClassificationMetrics
