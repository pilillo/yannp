package tokenizer

import (
	"errors"
)

// SpecialTokens represents special tokens used in tokenization
type SpecialTokens struct {
	PaddingToken   string
	EndOfSequence  string
	BeginningOfSeq string
	UnknownToken   string
	MaskToken      string
}

// DefaultSpecialTokens returns commonly used special tokens
func DefaultSpecialTokens() SpecialTokens {
	return SpecialTokens{
		PaddingToken:   "<|pad|>",
		EndOfSequence:  "<|endoftext|>",
		BeginningOfSeq: "<|startoftext|>",
		UnknownToken:   "<|unk|>",
		MaskToken:      "<|mask|>",
	}
}

// Tokenizer interface defines the contract for all tokenizers
type Tokenizer interface {
	// Encode converts text to token IDs
	Encode(text string) ([]int, error)

	// Decode converts token IDs to text
	Decode(tokenIds []int) (string, error)

	// GetVocabSize returns the vocabulary size
	GetVocabSize() int

	// GetSpecialTokens returns special tokens
	GetSpecialTokens() SpecialTokens

	// Save saves the tokenizer to files
	Save(vocabPath, mergePath string) error

	// Load loads the tokenizer from files
	Load(vocabPath, mergePath string) error
}

// Token represents a single token with its ID and text
type Token struct {
	ID   int
	Text string
}

// TokenizationResult contains the result of tokenization
type TokenizationResult struct {
	TokenIDs []int
	Tokens   []Token
	Text     string
}

var (
	ErrUnknownToken  = errors.New("unknown token")
	ErrInvalidInput  = errors.New("invalid input")
	ErrVocabMismatch = errors.New("vocabulary mismatch")
)
