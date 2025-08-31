package ann

import (
	"encoding/csv"
	"log"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var trainInputs *mat.Dense
var trainLabels *mat.Dense

var testInputs *mat.Dense
var testLabels *mat.Dense

func loadCSVData(filename string, numDataCols, numLabelCols int) (*mat.Dense, *mat.Dense, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = numDataCols + numLabelCols
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	inputsData := make([]float64, numDataCols*len(rawCSVData))
	labelsData := make([]float64, numLabelCols*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int
	for i, row := range rawCSVData {
		if i == 0 {
			continue
		}
		for j, val := range row {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, nil, err
			}
			if j >= numDataCols {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
			} else {
				inputsData[inputsIndex] = parsedVal
				inputsIndex++
			}
		}
	}

	return mat.NewDense(len(rawCSVData), numDataCols, inputsData), mat.NewDense(len(rawCSVData), numLabelCols, labelsData), nil
}

func TestMain(m *testing.M) {
	var err error
	trainInputs, trainLabels, err = loadCSVData("data/train.csv", 4, 3)
	if err != nil {
		log.Fatal(err)
	}
	testInputs, testLabels, err = loadCSVData("data/test.csv", 4, 3)
	if err != nil {
		log.Fatal(err)
	}
	exitVal := m.Run()
	os.Exit(exitVal)
}

func TestAnn(t *testing.T) {
	randomSeed := int64(123456)
	randSource := rand.NewSource(randomSeed)
	randGen := rand.New(randSource)

	mlp := NewMLP(4, 3, 3, NewSigmoidActivation())
	for i, _ := range mlp.layers {
		mlp.layers[i].Randomize(randGen)
	}

	mlp.Train(trainInputs, trainLabels, 5000, 0.3)

	predictions := mlp.Predict(testInputs)
	accuracy := NewAccuracyMetric().Assess(predictions, testLabels)

	assert.GreaterOrEqual(t, accuracy, 0.935)
}