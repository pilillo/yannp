package ann

type mlp struct {
	ann
}

func NewMLP(inputNeurons, hiddenNeurons, outputNeurons int, activationFunction ActivationFunction) *mlp {
	return &mlp{
		ann: ann{
			layers: []Layer{
				CreateLayer("hidden layer", inputNeurons, hiddenNeurons, &activationFunction),
				CreateLayer("output layer", hiddenNeurons, outputNeurons, &activationFunction),
			},
		},
	}
}