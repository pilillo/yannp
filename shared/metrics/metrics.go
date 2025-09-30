package metrics

import (
	"gonum.org/v1/gonum/mat"
)

type Metric interface {
	Assess(predictions, targets *mat.Dense) float64
}

type accuracyMetric struct{}

func NewAccuracyMetric() Metric {
	return &accuracyMetric{}
}

func (m *accuracyMetric) Assess(predictions, targets *mat.Dense) float64 {
	numCorrect := 0
	numSamples, _ := predictions.Dims()

	for i := 0; i < numSamples; i++ {
		predRow := mat.Row(nil, i, predictions)
		targetRow := mat.Row(nil, i, targets)

		maxPred := -1.0
		maxPredIndex := -1
		for j, val := range predRow {
			if val > maxPred {
				maxPred = val
				maxPredIndex = j
			}
		}

		maxTarget := -1.0
		maxTargetIndex := -1
		for j, val := range targetRow {
			if val > maxTarget {
				maxTarget = val
				maxTargetIndex = j
			}
		}

		if maxPredIndex == maxTargetIndex {
			numCorrect++
		}
	}

	return float64(numCorrect) / float64(numSamples)
}

func getLabels(predictions *mat.Dense) []int {
	numSamples, _ := predictions.Dims()
	labels := make([]int, numSamples)
	for i := 0; i < numSamples; i++ {
		row := mat.Row(nil, i, predictions)
		maxVal := -1.0
		maxIndex := -1
		for j, val := range row {
			if val > maxVal {
				maxVal = val
				maxIndex = j
			}
		}
		labels[i] = maxIndex
	}
	return labels
}

type ClassificationMetrics struct {
	Precision float64
	Recall    float64
	F1Score   float64
}

func NewClassificationMetrics(predictions, targets *mat.Dense) *ClassificationMetrics {
	predLabels := getLabels(predictions)
	targetLabels := getLabels(targets)

	_, numClasses := targets.Dims()

	tp := make([]int, numClasses)
	fp := make([]int, numClasses)
	fn := make([]int, numClasses)

	for i := 0; i < len(predLabels); i++ {
		pred := predLabels[i]
		target := targetLabels[i]

		if pred == target {
			tp[target]++
		} else {
			fp[pred]++
			fn[target]++
		}
	}

	precision := make([]float64, numClasses)
	recall := make([]float64, numClasses)
	f1 := make([]float64, numClasses)

	for i := 0; i < numClasses; i++ {
		if tp[i]+fp[i] > 0 {
			precision[i] = float64(tp[i]) / float64(tp[i]+fp[i])
		}
		if tp[i]+fn[i] > 0 {
			recall[i] = float64(tp[i]) / float64(tp[i]+fn[i])
		}
		if precision[i]+recall[i] > 0 {
			f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
		}
	}

	// Macro-average
	avgPrecision := 0.0
	avgRecall := 0.0
	avgF1 := 0.0
	for i := 0; i < numClasses; i++ {
		avgPrecision += precision[i]
		avgRecall += recall[i]
		avgF1 += f1[i]
	}

	return &ClassificationMetrics{
		Precision: avgPrecision / float64(numClasses),
		Recall:    avgRecall / float64(numClasses),
		F1Score:   avgF1 / float64(numClasses),
	}
}

func (m *ClassificationMetrics) Assess(predictions, targets *mat.Dense) float64 {
	return m.F1Score
}
