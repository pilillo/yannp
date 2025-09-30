package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// BPETokenizer implements Byte Pair Encoding tokenization
type BPETokenizer struct {
	vocab          map[string]int
	reverseVocab   map[int]string
	merges         map[string]int
	reverseMerges  map[int]string
	specialTokens  SpecialTokens
	vocabSize      int
	unknownTokenID int
	paddingTokenID int
	eosTokenID     int
	bosTokenID     int
	maskTokenID    int
}

// NewBPETokenizer creates a new BPE tokenizer
func NewBPETokenizer(specialTokens SpecialTokens) *BPETokenizer {
	tokenizer := &BPETokenizer{
		vocab:         make(map[string]int),
		reverseVocab:  make(map[int]string),
		merges:        make(map[string]int),
		reverseMerges: make(map[int]string),
		specialTokens: specialTokens,
	}

	// Add special tokens
	tokenizer.addSpecialTokens()

	return tokenizer
}

// Train trains the BPE tokenizer on a corpus
func (bpe *BPETokenizer) Train(corpus []string, vocabSize int) error {
	// Initialize with character-level vocabulary
	bpe.initCharacterVocab(corpus)

	// Count character pairs
	pairCounts := bpe.countPairs(corpus)

	// Perform BPE merges until vocab size is reached
	for len(bpe.vocab) < vocabSize {
		if len(pairCounts) == 0 {
			break
		}

		// Show progress every 50 merges
		if len(bpe.vocab)%50 == 0 {
			fmt.Printf("  📊 Vocab size: %d/%d (%.1f%%)\n",
				len(bpe.vocab), vocabSize,
				float64(len(bpe.vocab))/float64(vocabSize)*100)
		}

		// Find most frequent pair
		var bestPair string
		maxCount := 0
		for pair, count := range pairCounts {
			if count > maxCount {
				maxCount = count
				bestPair = pair
			}
		}

		if bestPair == "" || maxCount == 0 {
			break
		}

		// Add pair to vocabulary
		bpe.addMerge(bestPair)

		// Update corpus with new merge
		corpus = bpe.applyMergeToCorpus(corpus, bestPair)

		// Recount pairs
		pairCounts = bpe.countPairs(corpus)
	}

	bpe.vocabSize = len(bpe.vocab)
	return nil
}

// Encode converts text to token IDs using BPE
func (bpe *BPETokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return []int{}, nil
	}

	// Split into words (whitespace-separated)
	words := strings.Fields(text)
	var tokenIDs []int

	for _, word := range words {
		// Apply BPE to each word
		wordTokens := bpe.applyBPEToWord(word)

		for _, token := range wordTokens {
			if id, exists := bpe.vocab[token]; exists {
				tokenIDs = append(tokenIDs, id)
			} else {
				// Handle unknown tokens
				tokenIDs = append(tokenIDs, bpe.unknownTokenID)
			}
		}
	}

	return tokenIDs, nil
}

// Decode converts token IDs to text using BPE
func (bpe *BPETokenizer) Decode(tokenIds []int) (string, error) {
	var words []string
	var currentWord []string

	for _, tokenID := range tokenIds {
		token, exists := bpe.reverseVocab[tokenID]
		if !exists {
			return "", fmt.Errorf("unknown token ID: %d", tokenID)
		}

		if bpe.isSpecialToken(token) {
			// End current word if needed
			if len(currentWord) > 0 {
				words = append(words, strings.Join(currentWord, ""))
				currentWord = []string{}
			}
			continue
		}

		currentWord = append(currentWord, token)
	}

	// Add final word
	if len(currentWord) > 0 {
		words = append(words, strings.Join(currentWord, ""))
	}

	return strings.Join(words, " "), nil
}

// applyBPEToWord applies BPE to a single word
func (bpe *BPETokenizer) applyBPEToWord(word string) []string {
	// Add word boundaries
	tokens := bpe.wordToChars(word)

	// Apply merges
	for mergeID := len(bpe.reverseMerges); mergeID > 0; mergeID-- {
		merge, exists := bpe.reverseMerges[mergeID]
		if !exists {
			continue
		}

		// Split merge into parts
		parts := strings.Split(merge, " ")
		if len(parts) != 2 {
			continue
		}

		left, right := parts[0], parts[1]

		// Apply merge
		newTokens := []string{}
		i := 0
		for i < len(tokens) {
			if i < len(tokens)-1 && tokens[i] == left && tokens[i+1] == right {
				newTokens = append(newTokens, left+right)
				i += 2
			} else {
				newTokens = append(newTokens, tokens[i])
				i++
			}
		}
		tokens = newTokens
	}

	return tokens
}

// wordToChars converts a word to character-level tokens
func (bpe *BPETokenizer) wordToChars(word string) []string {
	var chars []string
	for _, r := range word {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			chars = append(chars, string(r))
		} else {
			chars = append(chars, string(r))
		}
	}
	return chars
}

// initCharacterVocab initializes vocabulary with characters
func (bpe *BPETokenizer) initCharacterVocab(corpus []string) {
	seen := make(map[string]bool)

	// Add special tokens
	bpe.addSpecialTokens()

	// Add characters from corpus
	for _, text := range corpus {
		for _, r := range text {
			char := string(r)
			if !seen[char] && !unicode.IsSpace(r) {
				bpe.addToken(char)
				seen[char] = true
			}
		}
	}
}

// countPairs counts character pairs in the corpus
func (bpe *BPETokenizer) countPairs(corpus []string) map[string]int {
	pairCounts := make(map[string]int)

	for _, text := range corpus {
		words := strings.Fields(text)
		for _, word := range words {
			tokens := bpe.applyBPEToWord(word)
			for i := 0; i < len(tokens)-1; i++ {
				pair := tokens[i] + " " + tokens[i+1]
				pairCounts[pair]++
			}
		}
	}

	return pairCounts
}

// addMerge adds a new merge to the tokenizer
func (bpe *BPETokenizer) addMerge(pair string) {
	mergeID := len(bpe.reverseMerges) + 1
	bpe.merges[pair] = mergeID
	bpe.reverseMerges[mergeID] = pair

	// Add merged token to vocabulary
	mergedToken := strings.Replace(pair, " ", "", 1)
	bpe.addToken(mergedToken)
}

// applyMergeToCorpus applies a merge to the entire corpus
func (bpe *BPETokenizer) applyMergeToCorpus(corpus []string, pair string) []string {
	left, right := strings.Split(pair, " ")[0], strings.Split(pair, " ")[1]
	merged := left + right

	var newCorpus []string
	for _, text := range corpus {
		newText := strings.Replace(text, left+" "+right, merged, -1)
		newCorpus = append(newCorpus, newText)
	}

	return newCorpus
}

// addToken adds a token to the vocabulary
func (bpe *BPETokenizer) addToken(token string) {
	if _, exists := bpe.vocab[token]; !exists {
		tokenID := len(bpe.vocab)
		bpe.vocab[token] = tokenID
		bpe.reverseVocab[tokenID] = token
	}
}

// addSpecialTokens adds special tokens to the vocabulary
func (bpe *BPETokenizer) addSpecialTokens() {
	tokens := []string{
		bpe.specialTokens.PaddingToken,
		bpe.specialTokens.EndOfSequence,
		bpe.specialTokens.BeginningOfSeq,
		bpe.specialTokens.UnknownToken,
		bpe.specialTokens.MaskToken,
	}

	for _, token := range tokens {
		bpe.addToken(token)
	}

	// Set special token IDs
	bpe.paddingTokenID = bpe.vocab[bpe.specialTokens.PaddingToken]
	bpe.eosTokenID = bpe.vocab[bpe.specialTokens.EndOfSequence]
	bpe.bosTokenID = bpe.vocab[bpe.specialTokens.BeginningOfSeq]
	bpe.unknownTokenID = bpe.vocab[bpe.specialTokens.UnknownToken]
	bpe.maskTokenID = bpe.vocab[bpe.specialTokens.MaskToken]
}

// isSpecialToken checks if a token is special
func (bpe *BPETokenizer) isSpecialToken(token string) bool {
	return token == bpe.specialTokens.PaddingToken ||
		token == bpe.specialTokens.EndOfSequence ||
		token == bpe.specialTokens.BeginningOfSeq ||
		token == bpe.specialTokens.UnknownToken ||
		token == bpe.specialTokens.MaskToken
}

// GetVocabSize returns the vocabulary size
func (bpe *BPETokenizer) GetVocabSize() int {
	return bpe.vocabSize
}

// GetSpecialTokens returns special tokens
func (bpe *BPETokenizer) GetSpecialTokens() SpecialTokens {
	return bpe.specialTokens
}

// Save saves the tokenizer to files
func (bpe *BPETokenizer) Save(vocabPath, mergePath string) error {
	// Save vocabulary
	vocabData := map[string]interface{}{
		"vocab":          bpe.vocab,
		"special_tokens": bpe.specialTokens,
	}

	vocabJSON, err := json.MarshalIndent(vocabData, "", "  ")
	if err != nil {
		return err
	}

	err = os.WriteFile(vocabPath, vocabJSON, 0644)
	if err != nil {
		return err
	}

	// Save merges
	mergeData := map[string]interface{}{
		"merges": bpe.merges,
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
func (bpe *BPETokenizer) Load(vocabPath, mergePath string) error {
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
				bpe.vocab[token] = int(idFloat)
				bpe.reverseVocab[int(idFloat)] = token
			}
		}
	}

	// Parse special tokens
	if specialTokensMap, ok := vocabFile["special_tokens"].(map[string]interface{}); ok {
		bpe.specialTokens = SpecialTokens{
			PaddingToken:   getString(specialTokensMap, "PaddingToken", ""),
			EndOfSequence:  getString(specialTokensMap, "EndOfSequence", ""),
			BeginningOfSeq: getString(specialTokensMap, "BeginningOfSeq", ""),
			UnknownToken:   getString(specialTokensMap, "UnknownToken", ""),
			MaskToken:      getString(specialTokensMap, "MaskToken", ""),
		}
	}

	// Load merges
	mergeData, err := os.ReadFile(mergePath)
	if err != nil {
		return err
	}

	var mergeFile map[string]interface{}
	err = json.Unmarshal(mergeData, &mergeFile)
	if err != nil {
		return err
	}

	if mergesMap, ok := mergeFile["merges"].(map[string]interface{}); ok {
		for pair, id := range mergesMap {
			if idFloat, ok := id.(float64); ok {
				bpe.merges[pair] = int(idFloat)
				bpe.reverseMerges[int(idFloat)] = pair
			}
		}
	}

	bpe.vocabSize = len(bpe.vocab)
	return nil
}

// Helper function to safely get string from map
func getString(m map[string]interface{}, key, defaultValue string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return defaultValue
}
