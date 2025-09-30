package main

import (
	"fmt"
	"math/rand"

	"github.com/pilillo/yannp/models/ann"
	"github.com/pilillo/yannp/shared/initialization"
	"github.com/pilillo/yannp/shared/loss"
	"github.com/pilillo/yannp/shared/optimizer"
	"github.com/pilillo/yannp/shared/regularization"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("🧠 YANNP Enhanced MLP Example")
	fmt.Println("============================")

	// Create sample data
	fmt.Println("📊 Creating sample dataset...")
	x := mat.NewDense(100, 4, nil)
	y := mat.NewDense(100, 3, nil)

	// Generate random data
	randGen := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		for j := 0; j < 4; j++ {
			x.Set(i, j, randGen.Float64())
		}
		// Create one-hot encoded labels
		label := randGen.Intn(3)
		for j := 0; j < 3; j++ {
			if j == label {
				y.Set(i, j, 1.0)
			} else {
				y.Set(i, j, 0.0)
			}
		}
	}

	// Create enhanced MLP
	fmt.Println("🏗️  Creating Enhanced MLP with 4 input, 8 hidden, 3 output neurons...")
	mlp := ann.NewEnhancedMLP(4, 8, 3, ann.NewReluActivation())

	// Configure with advanced features
	fmt.Println("⚙️  Configuring advanced features...")
	mlp.SetLossFunction(loss.NewCrossEntropy())
	mlp.SetOptimizerForAllLayers(optimizer.NewAdam(0.9, 0.999, 1e-8))
	mlp.SetWeightInitializerForAllLayers(initialization.NewHeNormal())
	mlp.SetRegularizerForAllLayers(regularization.NewL2Regularization(0.01))
	mlp.SetDropoutForLayer(0, regularization.NewDropout(0.2))
	mlp.SetBatchNormForLayer(0, regularization.NewBatchNormalization(8, 0.9, 1e-5))

	// Initialize
	fmt.Println("🎲 Initializing weights...")
	mlp.Initialize(randGen)

	// Create validation data
	fmt.Println("📋 Creating validation dataset...")
	valX := mat.NewDense(20, 4, nil)
	valY := mat.NewDense(20, 3, nil)

	// Generate validation data
	for i := 0; i < 20; i++ {
		for j := 0; j < 4; j++ {
			valX.Set(i, j, randGen.Float64())
		}
		label := randGen.Intn(3)
		for j := 0; j < 3; j++ {
			if j == label {
				valY.Set(i, j, 1.0)
			} else {
				valY.Set(i, j, 0.0)
			}
		}
	}

	// Train with validation
	fmt.Println("🎯 Training Enhanced MLP with validation...")
	valLosses := mlp.TrainWithValidation(x, y, valX, valY, 100, 0.001)

	fmt.Printf("📈 Final validation loss: %.4f\n", valLosses[len(valLosses)-1])

	// Predict
	fmt.Println("🧪 Testing the trained model...")
	predictions := mlp.Predict(valX)
	fmt.Println("Sample predictions (first 5 samples):")
	fmt.Println(mat.Formatted(predictions.Slice(0, 5, 0, 3), mat.Prefix("  ")))

	fmt.Println("\n✅ Enhanced MLP example completed!")
}
