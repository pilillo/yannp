package tokenizer

import (
	"regexp"
	"strings"
)

// BERTTokenizer provides BERT-compatible tokenization using WordPiece
type BERTTokenizer struct {
	wordpiece      *WordPieceTokenizer
	whitespaceRE   *regexp.Regexp
	punctuationRE  *regexp.Regexp
	basicTokenizer *BasicTokenizer
}

// BasicTokenizer handles basic text preprocessing for BERT
type BasicTokenizer struct {
	doLowerCase bool
}

// NewBERTTokenizer creates a new BERT-compatible tokenizer
func NewBERTTokenizer() *BERTTokenizer {
	specialTokens := SpecialTokens{
		PaddingToken:   "[PAD]",
		EndOfSequence:  "[SEP]",
		BeginningOfSeq: "[CLS]",
		UnknownToken:   "[UNK]",
		MaskToken:      "[MASK]",
	}

	return &BERTTokenizer{
		wordpiece:      NewWordPieceTokenizer(specialTokens),
		whitespaceRE:   regexp.MustCompile(`\s+`),
		punctuationRE:  regexp.MustCompile(`[.,!?;:()\[\]{}"'\x60]`),
		basicTokenizer: NewBasicTokenizer(true),
	}
}

// NewBasicTokenizer creates a new basic tokenizer
func NewBasicTokenizer(doLowerCase bool) *BasicTokenizer {
	return &BasicTokenizer{
		doLowerCase: doLowerCase,
	}
}

// Encode encodes text using BERT-style tokenization
func (bert *BERTTokenizer) Encode(text string) ([]int, error) {
	// Preprocess text
	text = bert.preprocess(text)

	// Basic tokenization
	tokens := bert.basicTokenizer.tokenize(text)

	// WordPiece tokenization
	var tokenIDs []int
	for _, token := range tokens {
		wordpieceTokens := bert.wordpiece.wordPieceTokenize(token)
		for _, wpToken := range wordpieceTokens {
			if id, exists := bert.wordpiece.vocab[wpToken]; exists {
				tokenIDs = append(tokenIDs, id)
			} else {
				tokenIDs = append(tokenIDs, bert.wordpiece.unknownTokenID)
			}
		}
	}

	return tokenIDs, nil
}

// Decode decodes token IDs to text
func (bert *BERTTokenizer) Decode(tokenIds []int) (string, error) {
	// Decode with WordPiece
	text, err := bert.wordpiece.Decode(tokenIds)
	if err != nil {
		return "", err
	}

	// Postprocess text
	text = bert.postprocess(text)

	return text, nil
}

// preprocess preprocesses text before tokenization
func (bert *BERTTokenizer) preprocess(text string) string {
	// Normalize whitespace
	text = bert.whitespaceRE.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// postprocess postprocesses text after decoding
func (bert *BERTTokenizer) postprocess(text string) string {
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
	text = bert.whitespaceRE.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// tokenize performs basic tokenization
func (bt *BasicTokenizer) tokenize(text string) []string {
	// Convert to lowercase if needed
	if bt.doLowerCase {
		text = strings.ToLower(text)
	}

	// Split on whitespace
	words := strings.Fields(text)

	// Further split on punctuation
	var tokens []string
	for _, word := range words {
		// Split on punctuation
		parts := bt.splitOnPunctuation(word)
		tokens = append(tokens, parts...)
	}

	return tokens
}

// splitOnPunctuation splits a word on punctuation
func (bt *BasicTokenizer) splitOnPunctuation(word string) []string {
	var parts []string
	var current strings.Builder

	for _, r := range word {
		if bt.isPunctuation(r) {
			if current.Len() > 0 {
				parts = append(parts, current.String())
				current.Reset()
			}
			parts = append(parts, string(r))
		} else {
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		parts = append(parts, current.String())
	}

	return parts
}

// isPunctuation checks if a rune is punctuation
func (bt *BasicTokenizer) isPunctuation(r rune) bool {
	return r == '.' || r == ',' || r == '!' || r == '?' || r == ';' || r == ':' ||
		r == '(' || r == ')' || r == '[' || r == ']' || r == '{' || r == '}' ||
		r == '"' || r == '\'' || r == '\x60'
}

// GetVocabSize returns vocabulary size
func (bert *BERTTokenizer) GetVocabSize() int {
	return bert.wordpiece.GetVocabSize()
}

// GetSpecialTokens returns special tokens
func (bert *BERTTokenizer) GetSpecialTokens() SpecialTokens {
	return bert.wordpiece.GetSpecialTokens()
}

// Save saves the tokenizer
func (bert *BERTTokenizer) Save(vocabPath, mergePath string) error {
	return bert.wordpiece.Save(vocabPath, mergePath)
}

// Load loads the tokenizer
func (bert *BERTTokenizer) Load(vocabPath, mergePath string) error {
	return bert.wordpiece.Load(vocabPath, mergePath)
}

// Train trains the tokenizer on a corpus
func (bert *BERTTokenizer) Train(corpus []string, vocabSize int) error {
	return bert.wordpiece.Train(corpus, vocabSize)
}
