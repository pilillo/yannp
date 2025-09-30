package optimizer

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Optimizer interface defines the contract for different optimization algorithms
type Optimizer interface {
	Update(weights, gradients *mat.Dense, learningRate float64, iteration int)
	GetName() string
}

// SGD implements Stochastic Gradient Descent
type SGD struct {
	momentum float64
	velocity *mat.Dense
}

func NewSGD(momentum float64) *SGD {
	return &SGD{
		momentum: momentum,
		velocity: nil,
	}
}

func (s *SGD) Update(weights, gradients *mat.Dense, learningRate float64, iteration int) {
	if s.velocity == nil {
		rows, cols := weights.Dims()
		s.velocity = mat.NewDense(rows, cols, nil)
	}

	// Update velocity with momentum
	s.velocity.Scale(s.momentum, s.velocity)
	var temp mat.Dense
	temp.Scale(learningRate, gradients)
	s.velocity.Add(s.velocity, &temp)

	// Update weights
	weights.Add(weights, s.velocity)
}

func (s *SGD) GetName() string {
	return "SGD"
}

// Adam implements the Adam optimizer
type Adam struct {
	beta1, beta2, epsilon float64
	m, v                  *mat.Dense
	t                     int
}

func NewAdam(beta1, beta2, epsilon float64) *Adam {
	return &Adam{
		beta1:   beta1,
		beta2:   beta2,
		epsilon: epsilon,
		m:       nil,
		v:       nil,
		t:       0,
	}
}

func (a *Adam) Update(weights, gradients *mat.Dense, learningRate float64, iteration int) {
	a.t++

	if a.m == nil {
		rows, cols := weights.Dims()
		a.m = mat.NewDense(rows, cols, nil)
		a.v = mat.NewDense(rows, cols, nil)
	}

	// Update biased first moment estimate
	a.m.Scale(a.beta1, a.m)
	var temp1 mat.Dense
	temp1.Scale(1-a.beta1, gradients)
	a.m.Add(a.m, &temp1)

	// Update biased second raw moment estimate
	a.v.Scale(a.beta2, a.v)
	var temp2 mat.Dense
	temp2.MulElem(gradients, gradients)
	temp2.Scale(1-a.beta2, &temp2)
	a.v.Add(a.v, &temp2)

	// Compute bias-corrected first moment estimate
	var mHat mat.Dense
	mHat.Scale(1/(1-math.Pow(a.beta1, float64(a.t))), a.m)

	// Compute bias-corrected second raw moment estimate
	var vHat mat.Dense
	vHat.Scale(1/(1-math.Pow(a.beta2, float64(a.t))), a.v)

	// Update weights
	var update mat.Dense
	update.Apply(func(i, j int, v float64) float64 {
		return learningRate * mHat.At(i, j) / (math.Sqrt(vHat.At(i, j)) + a.epsilon)
	}, &vHat)

	weights.Add(weights, &update)
}

func (a *Adam) GetName() string {
	return "Adam"
}

// RMSprop implements the RMSprop optimizer
type RMSprop struct {
	decay, epsilon float64
	cache          *mat.Dense
}

func NewRMSprop(decay, epsilon float64) *RMSprop {
	return &RMSprop{
		decay:   decay,
		epsilon: epsilon,
		cache:   nil,
	}
}

func (r *RMSprop) Update(weights, gradients *mat.Dense, learningRate float64, iteration int) {
	if r.cache == nil {
		rows, cols := weights.Dims()
		r.cache = mat.NewDense(rows, cols, nil)
	}

	// Update cache with decay
	r.cache.Scale(r.decay, r.cache)
	var temp mat.Dense
	temp.MulElem(gradients, gradients)
	temp.Scale(1-r.decay, &temp)
	r.cache.Add(r.cache, &temp)

	// Update weights
	var update mat.Dense
	update.Apply(func(i, j int, v float64) float64 {
		return learningRate * gradients.At(i, j) / (math.Sqrt(r.cache.At(i, j)) + r.epsilon)
	}, r.cache)

	weights.Add(weights, &update)
}

func (r *RMSprop) GetName() string {
	return "RMSprop"
}

// Adagrad implements the Adagrad optimizer
type Adagrad struct {
	epsilon float64
	cache   *mat.Dense
}

func NewAdagrad(epsilon float64) *Adagrad {
	return &Adagrad{
		epsilon: epsilon,
		cache:   nil,
	}
}

func (a *Adagrad) Update(weights, gradients *mat.Dense, learningRate float64, iteration int) {
	if a.cache == nil {
		rows, cols := weights.Dims()
		a.cache = mat.NewDense(rows, cols, nil)
	}

	// Update cache
	var temp mat.Dense
	temp.MulElem(gradients, gradients)
	a.cache.Add(a.cache, &temp)

	// Update weights
	var update mat.Dense
	update.Apply(func(i, j int, v float64) float64 {
		return learningRate * gradients.At(i, j) / (math.Sqrt(a.cache.At(i, j)) + a.epsilon)
	}, a.cache)

	weights.Add(weights, &update)
}

func (a *Adagrad) GetName() string {
	return "Adagrad"
}
