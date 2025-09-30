package main

import (
	"fmt"
	"math/rand"

	"github.com/pilillo/yannp/models/ann"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("🧠 YANNP Basic MLP Example (XOR Problem)")
	fmt.Println("========================================")

	// Create sample data for XOR problem
	fmt.Println("📊 Creating XOR dataset...")
	x := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})

	y := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})

	fmt.Println("Input data:")
	fmt.Println(mat.Formatted(x, mat.Prefix("  ")))
	fmt.Println("\nTarget data:")
	fmt.Println(mat.Formatted(y, mat.Prefix("  ")))

	// Create MLP
	fmt.Println("\n🏗️  Creating Basic MLP with 2 input, 4 hidden, 1 output neurons...")
	mlp := ann.NewMLP(2, 4, 1, ann.NewSigmoidActivation())

	// Initialize weights
	fmt.Println("🎲 Initializing weights with random values...")
	randGen := rand.New(rand.NewSource(42))
	mlp.Initialize(randGen)

	fmt.Println("📏 Network architecture:")
	fmt.Println("  - Input layer: 2 neurons")
	fmt.Println("  - Hidden layer: 4 neurons (Sigmoid activation)")
	fmt.Println("  - Output layer: 1 neuron (Sigmoid activation)")

	// Train
	fmt.Println("\n🎯 Training MLP...")
	fmt.Println("  - Epochs: 1000")
	fmt.Println("  - Learning rate: 0.5")
	mlp.Train(x, y, 1000, 0.5)

	// Predict
	fmt.Println("\n🧪 Testing the trained model...")
	predictions := mlp.Predict(x)
	fmt.Println("Predictions:")
	fmt.Println(mat.Formatted(predictions, mat.Prefix("  ")))

	// Calculate and show accuracy
	fmt.Println("\n📊 Evaluating results:")
	correct := 0
	for i := 0; i < 4; i++ {
		pred := predictions.At(i, 0)
		target := y.At(i, 0)

		predRounded := 0.0
		if pred > 0.5 {
			predRounded = 1.0
		}

		if predRounded == target {
			correct++
		}

		fmt.Printf("  Input: [%.0f, %.0f] -> Predicted: %.4f (%.0f) | Target: %.0f | %s\n",
			x.At(i, 0), x.At(i, 1), pred, predRounded, target,
			func() string {
				if predRounded == target {
					return "✅"
				}
				return "❌"
			}())
	}

	accuracy := float64(correct) / 4.0 * 100
	fmt.Printf("\n📈 Accuracy: %.1f%% (%d/4 correct)\n", accuracy, correct)

	fmt.Println("\n✅ Basic MLP example completed!")
	fmt.Println("\n💡 Note: The XOR problem is a classic non-linearly separable problem")
	fmt.Println("   that requires a hidden layer to solve successfully.")
}
