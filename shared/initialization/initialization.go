package initialization

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// WeightInitializer interface defines the contract for different weight initialization strategies
type WeightInitializer interface {
	Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand)
	GetName() string
}

// XavierUniform implements Xavier/Glorot uniform initialization
type XavierUniform struct{}

func NewXavierUniform() *XavierUniform {
	return &XavierUniform{}
}

func (x *XavierUniform) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	limit := math.Sqrt(6.0 / float64(inputSize+outputSize))

	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.Float64()*2*limit - limit
	}, weights)
}

func (x *XavierUniform) GetName() string {
	return "XavierUniform"
}

// XavierNormal implements Xavier/Glorot normal initialization
type XavierNormal struct{}

func NewXavierNormal() *XavierNormal {
	return &XavierNormal{}
}

func (x *XavierNormal) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	std := math.Sqrt(2.0 / float64(inputSize+outputSize))

	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * std
	}, weights)
}

func (x *XavierNormal) GetName() string {
	return "XavierNormal"
}

// HeUniform implements He uniform initialization (for ReLU networks)
type HeUniform struct{}

func NewHeUniform() *HeUniform {
	return &HeUniform{}
}

func (h *HeUniform) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	limit := math.Sqrt(6.0 / float64(inputSize))

	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.Float64()*2*limit - limit
	}, weights)
}

func (h *HeUniform) GetName() string {
	return "HeUniform"
}

// HeNormal implements He normal initialization (for ReLU networks)
type HeNormal struct{}

func NewHeNormal() *HeNormal {
	return &HeNormal{}
}

func (h *HeNormal) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	std := math.Sqrt(2.0 / float64(inputSize))

	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * std
	}, weights)
}

func (h *HeNormal) GetName() string {
	return "HeNormal"
}

// LeCunUniform implements LeCun uniform initialization
type LeCunUniform struct{}

func NewLeCunUniform() *LeCunUniform {
	return &LeCunUniform{}
}

func (l *LeCunUniform) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	limit := math.Sqrt(3.0 / float64(inputSize))

	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.Float64()*2*limit - limit
	}, weights)
}

func (l *LeCunUniform) GetName() string {
	return "LeCunUniform"
}

// LeCunNormal implements LeCun normal initialization
type LeCunNormal struct{}

func NewLeCunNormal() *LeCunNormal {
	return &LeCunNormal{}
}

func (l *LeCunNormal) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	std := math.Sqrt(1.0 / float64(inputSize))

	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64() * std
	}, weights)
}

func (l *LeCunNormal) GetName() string {
	return "LeCunNormal"
}

// RandomUniform implements simple uniform initialization
type RandomUniform struct {
	Min, Max float64
}

func NewRandomUniform(min, max float64) *RandomUniform {
	return &RandomUniform{Min: min, Max: max}
}

func (r *RandomUniform) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.Float64()*(r.Max-r.Min) + r.Min
	}, weights)
}

func (r *RandomUniform) GetName() string {
	return "RandomUniform"
}

// RandomNormal implements simple normal initialization
type RandomNormal struct {
	Mean, Std float64
}

func NewRandomNormal(mean, std float64) *RandomNormal {
	return &RandomNormal{Mean: mean, Std: std}
}

func (r *RandomNormal) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	weights.Apply(func(i, j int, v float64) float64 {
		return randGen.NormFloat64()*r.Std + r.Mean
	}, weights)
}

func (r *RandomNormal) GetName() string {
	return "RandomNormal"
}

// Zeros implements zero initialization
type Zeros struct{}

func NewZeros() *Zeros {
	return &Zeros{}
}

func (z *Zeros) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	weights.Apply(func(i, j int, v float64) float64 {
		return 0.0
	}, weights)
}

func (z *Zeros) GetName() string {
	return "Zeros"
}

// Ones implements ones initialization
type Ones struct{}

func NewOnes() *Ones {
	return &Ones{}
}

func (o *Ones) Initialize(weights *mat.Dense, inputSize, outputSize int, randGen *rand.Rand) {
	weights.Apply(func(i, j int, v float64) float64 {
		return 1.0
	}, weights)
}

func (o *Ones) GetName() string {
	return "Ones"
}
