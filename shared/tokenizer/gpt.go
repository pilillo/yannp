package tokenizer

import (
	"regexp"
	"strings"
)

// GPTTokenizer provides GPT-compatible tokenization
type GPTTokenizer struct {
	bpe           *BPETokenizer
	tiktoken      *TiktokenWrapper
	whitespaceRE  *regexp.Regexp
	punctuationRE *regexp.Regexp
}

// NewGPTTokenizer creates a new GPT-compatible tokenizer
func NewGPTTokenizer() *GPTTokenizer {
	specialTokens := SpecialTokens{
		PaddingToken:   "<|endoftext|>",
		EndOfSequence:  "<|endoftext|>",
		BeginningOfSeq: "<|endoftext|>",
		UnknownToken:   "<|endoftext|>",
		MaskToken:      "<|mask|>",
	}

	return &GPTTokenizer{
		bpe:           NewBPETokenizer(specialTokens),
		whitespaceRE:  regexp.MustCompile(`\s+`),
		punctuationRE: regexp.MustCompile(`[.,!?;:()\[\]{}"'\x60]`),
	}
}

// Encode encodes text using GPT-style tokenization
func (gpt *GPTTokenizer) Encode(text string) ([]int, error) {
	// Use tiktoken if available, otherwise fall back to BPE
	if gpt.tiktoken != nil {
		return gpt.tiktoken.Encode(text)
	}

	// Preprocess text
	text = gpt.preprocess(text)

	// Encode with BPE
	tokenIDs, err := gpt.bpe.Encode(text)
	if err != nil {
		return nil, err
	}

	return tokenIDs, nil
}

// Decode decodes token IDs to text
func (gpt *GPTTokenizer) Decode(tokenIds []int) (string, error) {
	// Use tiktoken if available, otherwise fall back to BPE
	if gpt.tiktoken != nil {
		return gpt.tiktoken.Decode(tokenIds)
	}

	// Decode with BPE
	text, err := gpt.bpe.Decode(tokenIds)
	if err != nil {
		return "", err
	}

	// Postprocess text
	text = gpt.postprocess(text)

	return text, nil
}

// preprocess preprocesses text before tokenization
func (gpt *GPTTokenizer) preprocess(text string) string {
	// Normalize whitespace
	text = gpt.whitespaceRE.ReplaceAllString(text, " ")

	// Add spaces around punctuation
	text = gpt.punctuationRE.ReplaceAllStringFunc(text, func(match string) string {
		return " " + match + " "
	})

	return strings.TrimSpace(text)
}

// postprocess postprocesses text after decoding
func (gpt *GPTTokenizer) postprocess(text string) string {
	// Remove extra spaces around punctuation
	text = strings.ReplaceAll(text, " .", ".")
	text = strings.ReplaceAll(text, " ,", ",")
	text = strings.ReplaceAll(text, " !", "!")
	text = strings.ReplaceAll(text, " ?", "?")
	text = strings.ReplaceAll(text, " ;", ";")
	text = strings.ReplaceAll(text, " :", ":")
	text = strings.ReplaceAll(text, " )", ")")
	text = strings.ReplaceAll(text, "( ", "(")
	text = strings.ReplaceAll(text, " ]", "]")
	text = strings.ReplaceAll(text, "[ ", "[")
	text = strings.ReplaceAll(text, " }", "}")
	text = strings.ReplaceAll(text, "{ ", "{")

	// Normalize whitespace
	text = gpt.whitespaceRE.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// GetVocabSize returns vocabulary size
func (gpt *GPTTokenizer) GetVocabSize() int {
	return gpt.bpe.GetVocabSize()
}

// GetSpecialTokens returns special tokens
func (gpt *GPTTokenizer) GetSpecialTokens() SpecialTokens {
	return gpt.bpe.GetSpecialTokens()
}

// Save saves the tokenizer
func (gpt *GPTTokenizer) Save(vocabPath, mergePath string) error {
	return gpt.bpe.Save(vocabPath, mergePath)
}

// Load loads the tokenizer
func (gpt *GPTTokenizer) Load(vocabPath, mergePath string) error {
	return gpt.bpe.Load(vocabPath, mergePath)
}

// Train trains the tokenizer on a corpus
func (gpt *GPTTokenizer) Train(corpus []string, vocabSize int) error {
	return gpt.bpe.Train(corpus, vocabSize)
}

// SetTiktokenWrapper sets the tiktoken wrapper for this tokenizer
func (gpt *GPTTokenizer) SetTiktokenWrapper(tiktoken *TiktokenWrapper) {
	gpt.tiktoken = tiktoken
}
