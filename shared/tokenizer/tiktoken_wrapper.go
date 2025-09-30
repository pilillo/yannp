package tokenizer

import (
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
)

// TiktokenWrapper provides a wrapper around tiktoken-go for GPT models
type TiktokenWrapper struct {
	tiktoken *tiktoken.Tiktoken
	model    string
}

// NewTiktokenWrapper creates a new tiktoken wrapper
func NewTiktokenWrapper(model string) (*TiktokenWrapper, error) {
	// Map model names to tiktoken encodings
	var encodingName string
	switch {
	case strings.Contains(model, "gpt-4o") || strings.Contains(model, "gpt-4.1") || strings.Contains(model, "gpt-4.5"):
		encodingName = "o200k_base"
	case strings.Contains(model, "gpt-4") || strings.Contains(model, "gpt-3.5-turbo"):
		encodingName = "cl100k_base"
	case strings.Contains(model, "code-davinci") || strings.Contains(model, "text-davinci"):
		encodingName = "p50k_base"
	case strings.Contains(model, "gpt2") || strings.Contains(model, "gpt-3"):
		encodingName = "r50k_base"
	default:
		// Default to GPT-2 encoding for our demo
		encodingName = "gpt2"
	}

	tiktokenInstance, err := tiktoken.GetEncoding(encodingName)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding %s: %v", encodingName, err)
	}

	return &TiktokenWrapper{
		tiktoken: tiktokenInstance,
		model:    model,
	}, nil
}

// Encode encodes text to tokens
func (tw *TiktokenWrapper) Encode(text string) ([]int, error) {
	return tw.tiktoken.Encode(text, nil, nil), nil
}

// Decode decodes tokens to text
func (tw *TiktokenWrapper) Decode(tokens []int) (string, error) {
	return tw.tiktoken.Decode(tokens), nil
}

// GetVocabSize returns the vocabulary size
func (tw *TiktokenWrapper) GetVocabSize() int {
	// Return appropriate vocab size based on encoding
	switch tw.GetEncodingName() {
	case "r50k_base":
		return 50257
	case "p50k_base":
		return 50280
	case "cl100k_base":
		return 100256
	case "o200k_base":
		return 200256
	default:
		return 50257 // Default to GPT-2 size
	}
}

// GetModel returns the model name
func (tw *TiktokenWrapper) GetModel() string {
	return tw.model
}

// GetEncodingName returns the tiktoken encoding name
func (tw *TiktokenWrapper) GetEncodingName() string {
	switch {
	case strings.Contains(tw.model, "gpt-4o") || strings.Contains(tw.model, "gpt-4.1") || strings.Contains(tw.model, "gpt-4.5"):
		return "o200k_base"
	case strings.Contains(tw.model, "gpt-4") || strings.Contains(tw.model, "gpt-3.5-turbo"):
		return "cl100k_base"
	case strings.Contains(tw.model, "code-davinci") || strings.Contains(tw.model, "text-davinci"):
		return "p50k_base"
	case strings.Contains(tw.model, "gpt2") || strings.Contains(tw.model, "gpt-3"):
		return "r50k_base"
	default:
		return "gpt2"
	}
}
