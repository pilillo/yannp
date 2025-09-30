package tokenizer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode"
)

// SentencePieceTokenizer implements SentencePiece tokenization
// This is a simplified implementation - full SentencePiece is more complex
type SentencePieceTokenizer struct {
	vocab             map[string]int
	reverseVocab      map[int]string
	specialTokens     SpecialTokens
	vocabSize         int
	unknownTokenID    int
	paddingTokenID    int
	eosTokenID        int
	bosTokenID        int
	maskTokenID       int
	unkToken          string
	characterCoverage float64
	modelType         string
}

// NewSentencePieceTokenizer creates a new SentencePiece tokenizer
func NewSentencePieceTokenizer(specialTokens SpecialTokens) *SentencePieceTokenizer {
	tokenizer := &SentencePieceTokenizer{
		vocab:             make(map[string]int),
		reverseVocab:      make(map[int]string),
		specialTokens:     specialTokens,
		characterCoverage: 0.9995,
		modelType:         "unigram",
		unkToken:          specialTokens.UnknownToken,
	}

	// Add special tokens
	tokenizer.addSpecialTokens()

	return tokenizer
}

// Train trains the SentencePiece tokenizer on a corpus
func (sp *SentencePieceTokenizer) Train(corpus []string, vocabSize int) error {
	// Initialize with character-level vocabulary
	sp.initCharacterVocab(corpus)

	// Count character frequencies
	charCounts := sp.countCharacters(corpus)

	// Initialize vocabulary with characters and special tokens
	sp.initVocab()

	// Perform SentencePiece training (simplified unigram model)
	for len(sp.vocab) < vocabSize {
		// Find best subwords to add
		bestSubwords := sp.findBestSubwords(corpus, charCounts)

		if len(bestSubwords) == 0 {
			break
		}

		// Add the best subwords to vocabulary
		for _, subword := range bestSubwords {
			if len(sp.vocab) >= vocabSize {
				break
			}
			sp.addToken(subword)
		}
	}

	sp.vocabSize = len(sp.vocab)
	return nil
}

// Encode converts text to token IDs using SentencePiece
func (sp *SentencePieceTokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return []int{}, nil
	}

	// Normalize text
	text = sp.normalizeText(text)

	// Apply SentencePiece tokenization
	tokens := sp.tokenize(text)

	var tokenIDs []int
	for _, token := range tokens {
		if id, exists := sp.vocab[token]; exists {
			tokenIDs = append(tokenIDs, id)
		} else {
			// Handle unknown tokens
			tokenIDs = append(tokenIDs, sp.unknownTokenID)
		}
	}

	return tokenIDs, nil
}

// Decode converts token IDs to text using SentencePiece
func (sp *SentencePieceTokenizer) Decode(tokenIds []int) (string, error) {
	var tokens []string

	for _, tokenID := range tokenIds {
		token, exists := sp.reverseVocab[tokenID]
		if !exists {
			return "", fmt.Errorf("unknown token ID: %d", tokenID)
		}

		if sp.isSpecialToken(token) {
			continue
		}

		tokens = append(tokens, token)
	}

	// Join tokens and denormalize
	text := strings.Join(tokens, "")
	text = sp.denormalizeText(text)

	return text, nil
}

// tokenize tokenizes text using SentencePiece
func (sp *SentencePieceTokenizer) tokenize(text string) []string {
	// This is a simplified implementation
	// Real SentencePiece uses more sophisticated algorithms

	// Split into characters first
	chars := []rune(text)
	var tokens []string

	i := 0
	for i < len(chars) {
		// Try to find the longest subword starting at position i
		longestSubword := ""
		longestLength := 0

		for j := i + 1; j <= len(chars); j++ {
			subword := string(chars[i:j])
			if _, exists := sp.vocab[subword]; exists {
				if len(subword) > longestLength {
					longestSubword = subword
					longestLength = len(subword)
				}
			}
		}

		if longestSubword != "" {
			tokens = append(tokens, longestSubword)
			i += longestLength
		} else {
			// Single character
			tokens = append(tokens, string(chars[i]))
			i++
		}
	}

	return tokens
}

// normalizeText normalizes text before tokenization
func (sp *SentencePieceTokenizer) normalizeText(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Normalize whitespace
	text = strings.Join(strings.Fields(text), " ")

	// Add spaces around punctuation
	text = strings.ReplaceAll(text, ".", " .")
	text = strings.ReplaceAll(text, ",", " ,")
	text = strings.ReplaceAll(text, "!", " !")
	text = strings.ReplaceAll(text, "?", " ?")
	text = strings.ReplaceAll(text, ";", " ;")
	text = strings.ReplaceAll(text, ":", " :")
	text = strings.ReplaceAll(text, "(", " ( ")
	text = strings.ReplaceAll(text, ")", " ) ")
	text = strings.ReplaceAll(text, "[", " [ ")
	text = strings.ReplaceAll(text, "]", " ] ")
	text = strings.ReplaceAll(text, "{", " { ")
	text = strings.ReplaceAll(text, "}", " } ")

	return strings.TrimSpace(text)
}

// denormalizeText denormalizes text after decoding
func (sp *SentencePieceTokenizer) denormalizeText(text string) string {
	// Remove extra spaces around punctuation
	text = strings.ReplaceAll(text, " .", ".")
	text = strings.ReplaceAll(text, " ,", ",")
	text = strings.ReplaceAll(text, " !", "!")
	text = strings.ReplaceAll(text, " ?", "?")
	text = strings.ReplaceAll(text, " ;", ";")
	text = strings.ReplaceAll(text, " :", ":")
	text = strings.ReplaceAll(text, " ( ", "(")
	text = strings.ReplaceAll(text, " ) ", ")")
	text = strings.ReplaceAll(text, " [ ", "[")
	text = strings.ReplaceAll(text, " ] ", "]")
	text = strings.ReplaceAll(text, " { ", "{")
	text = strings.ReplaceAll(text, " } ", "}")

	// Normalize whitespace
	text = strings.Join(strings.Fields(text), " ")

	return text
}

// initCharacterVocab initializes vocabulary with characters
func (sp *SentencePieceTokenizer) initCharacterVocab(corpus []string) {
	seen := make(map[string]bool)

	// Add special tokens
	sp.addSpecialTokens()

	// Add characters from corpus
	for _, text := range corpus {
		text = sp.normalizeText(text)
		for _, r := range text {
			char := string(r)
			if !seen[char] && !unicode.IsSpace(r) {
				sp.addToken(char)
				seen[char] = true
			}
		}
	}
}

// countCharacters counts character frequencies in the corpus
func (sp *SentencePieceTokenizer) countCharacters(corpus []string) map[string]int {
	charCounts := make(map[string]int)

	for _, text := range corpus {
		text = sp.normalizeText(text)
		for _, r := range text {
			char := string(r)
			if !unicode.IsSpace(r) {
				charCounts[char]++
			}
		}
	}

	return charCounts
}

// initVocab initializes the vocabulary
func (sp *SentencePieceTokenizer) initVocab() {
	// Vocabulary is already initialized with characters and special tokens
}

// findBestSubwords finds the best subwords to add to vocabulary
func (sp *SentencePieceTokenizer) findBestSubwords(corpus []string, charCounts map[string]int) []string {
	type subwordScore struct {
		subword string
		score   float64
	}

	var scores []subwordScore

	// For each text, find potential subwords
	for _, text := range corpus {
		text = sp.normalizeText(text)
		chars := []rune(text)

		// Try all possible subwords of length 2 to 4
		for length := 2; length <= 4 && length <= len(chars); length++ {
			for i := 0; i <= len(chars)-length; i++ {
				subword := string(chars[i : i+length])

				// Calculate score based on frequency and length
				score := float64(charCounts[subword]) * math.Log(float64(len(subword)))

				if score > 0 {
					scores = append(scores, subwordScore{
						subword: subword,
						score:   score,
					})
				}
			}
		}
	}

	// Sort by score (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return unique subwords
	seen := make(map[string]bool)
	var result []string
	for _, score := range scores {
		if !seen[score.subword] && len(result) < 100 { // Limit to top 100
			result = append(result, score.subword)
			seen[score.subword] = true
		}
	}

	return result
}

// addToken adds a token to the vocabulary
func (sp *SentencePieceTokenizer) addToken(token string) {
	if _, exists := sp.vocab[token]; !exists {
		tokenID := len(sp.vocab)
		sp.vocab[token] = tokenID
		sp.reverseVocab[tokenID] = token
	}
}

// addSpecialTokens adds special tokens to the vocabulary
func (sp *SentencePieceTokenizer) addSpecialTokens() {
	tokens := []string{
		sp.specialTokens.PaddingToken,
		sp.specialTokens.EndOfSequence,
		sp.specialTokens.BeginningOfSeq,
		sp.specialTokens.UnknownToken,
		sp.specialTokens.MaskToken,
	}

	for _, token := range tokens {
		sp.addToken(token)
	}

	// Set special token IDs
	sp.paddingTokenID = sp.vocab[sp.specialTokens.PaddingToken]
	sp.eosTokenID = sp.vocab[sp.specialTokens.EndOfSequence]
	sp.bosTokenID = sp.vocab[sp.specialTokens.BeginningOfSeq]
	sp.unknownTokenID = sp.vocab[sp.specialTokens.UnknownToken]
	sp.maskTokenID = sp.vocab[sp.specialTokens.MaskToken]
}

// isSpecialToken checks if a token is special
func (sp *SentencePieceTokenizer) isSpecialToken(token string) bool {
	return token == sp.specialTokens.PaddingToken ||
		token == sp.specialTokens.EndOfSequence ||
		token == sp.specialTokens.BeginningOfSeq ||
		token == sp.specialTokens.UnknownToken ||
		token == sp.specialTokens.MaskToken
}

// GetVocabSize returns the vocabulary size
func (sp *SentencePieceTokenizer) GetVocabSize() int {
	return sp.vocabSize
}

// GetSpecialTokens returns special tokens
func (sp *SentencePieceTokenizer) GetSpecialTokens() SpecialTokens {
	return sp.specialTokens
}

// Save saves the tokenizer to files
func (sp *SentencePieceTokenizer) Save(vocabPath, mergePath string) error {
	// Save vocabulary
	vocabData := map[string]interface{}{
		"vocab":              sp.vocab,
		"special_tokens":     sp.specialTokens,
		"model_type":         sp.modelType,
		"character_coverage": sp.characterCoverage,
	}

	vocabJSON, err := json.MarshalIndent(vocabData, "", "  ")
	if err != nil {
		return err
	}

	err = os.WriteFile(vocabPath, vocabJSON, 0644)
	if err != nil {
		return err
	}

	// SentencePiece doesn't use merges, so create empty file
	mergeData := map[string]interface{}{
		"merges": map[string]int{},
	}

	mergeJSON, err := json.MarshalIndent(mergeData, "", "  ")
	if err != nil {
		return err
	}

	err = os.WriteFile(mergePath, mergeJSON, 0644)
	if err != nil {
		return err
	}

	return nil
}

// Load loads the tokenizer from files
func (sp *SentencePieceTokenizer) Load(vocabPath, mergePath string) error {
	// Load vocabulary
	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		return err
	}

	var vocabFile map[string]interface{}
	err = json.Unmarshal(vocabData, &vocabFile)
	if err != nil {
		return err
	}

	// Parse vocabulary
	if vocabMap, ok := vocabFile["vocab"].(map[string]interface{}); ok {
		for token, id := range vocabMap {
			if idFloat, ok := id.(float64); ok {
				sp.vocab[token] = int(idFloat)
				sp.reverseVocab[int(idFloat)] = token
			}
		}
	}

	// Parse special tokens
	if specialTokensMap, ok := vocabFile["special_tokens"].(map[string]interface{}); ok {
		sp.specialTokens = SpecialTokens{
			PaddingToken:   getString(specialTokensMap, "PaddingToken", ""),
			EndOfSequence:  getString(specialTokensMap, "EndOfSequence", ""),
			BeginningOfSeq: getString(specialTokensMap, "BeginningOfSeq", ""),
			UnknownToken:   getString(specialTokensMap, "UnknownToken", ""),
			MaskToken:      getString(specialTokensMap, "MaskToken", ""),
		}
	}

	// Parse model type
	if modelType, ok := vocabFile["model_type"].(string); ok {
		sp.modelType = modelType
	}

	// Parse character coverage
	if coverage, ok := vocabFile["character_coverage"].(float64); ok {
		sp.characterCoverage = coverage
	}

	sp.vocabSize = len(sp.vocab)
	return nil
}
