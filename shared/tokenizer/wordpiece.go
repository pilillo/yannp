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

// WordPieceTokenizer implements WordPiece tokenization (used by BERT)
type WordPieceTokenizer struct {
	vocab          map[string]int
	reverseVocab   map[int]string
	specialTokens  SpecialTokens
	vocabSize      int
	unknownTokenID int
	paddingTokenID int
	eosTokenID     int
	bosTokenID     int
	maskTokenID    int
	maxInputChars  int
	unkToken       string
}

// NewWordPieceTokenizer creates a new WordPiece tokenizer
func NewWordPieceTokenizer(specialTokens SpecialTokens) *WordPieceTokenizer {
	tokenizer := &WordPieceTokenizer{
		vocab:         make(map[string]int),
		reverseVocab:  make(map[int]string),
		specialTokens: specialTokens,
		maxInputChars: 200,
		unkToken:      specialTokens.UnknownToken,
	}

	// Add special tokens
	tokenizer.addSpecialTokens()

	return tokenizer
}

// Train trains the WordPiece tokenizer on a corpus
func (wp *WordPieceTokenizer) Train(corpus []string, vocabSize int) error {
	// Initialize with character-level vocabulary
	wp.initCharacterVocab(corpus)

	// Count word frequencies
	wordCounts := wp.countWords(corpus)

	// Add all words to vocabulary initially
	for word := range wordCounts {
		wp.addToken(word)
	}

	// Perform WordPiece training
	for len(wp.vocab) < vocabSize {
		// Find best split for each word
		bestSplits := wp.findBestSplits(wordCounts)

		if len(bestSplits) == 0 {
			break
		}

		// Add the best split to vocabulary
		bestSplit := bestSplits[0]
		wp.addToken(bestSplit)

		// Update word counts with new split
		wordCounts = wp.updateWordCounts(wordCounts, bestSplit)
	}

	wp.vocabSize = len(wp.vocab)
	return nil
}

// Encode converts text to token IDs using WordPiece
func (wp *WordPieceTokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return []int{}, nil
	}

	// Split into words
	words := strings.Fields(text)
	var tokenIDs []int

	for _, word := range words {
		// Apply WordPiece to each word
		wordTokens := wp.wordPieceTokenize(word)

		for _, token := range wordTokens {
			if id, exists := wp.vocab[token]; exists {
				tokenIDs = append(tokenIDs, id)
			} else {
				// Handle unknown tokens
				tokenIDs = append(tokenIDs, wp.unknownTokenID)
			}
		}
	}

	return tokenIDs, nil
}

// Decode converts token IDs to text using WordPiece
func (wp *WordPieceTokenizer) Decode(tokenIds []int) (string, error) {
	var words []string
	var currentWord []string

	for _, tokenID := range tokenIds {
		token, exists := wp.reverseVocab[tokenID]
		if !exists {
			return "", fmt.Errorf("unknown token ID: %d", tokenID)
		}

		if wp.isSpecialToken(token) {
			// End current word if needed
			if len(currentWord) > 0 {
				words = append(words, strings.Join(currentWord, ""))
				currentWord = []string{}
			}
			continue
		}

		// Check if token starts with ## (subword continuation)
		if strings.HasPrefix(token, "##") {
			// Remove ## prefix and add to current word
			subword := strings.TrimPrefix(token, "##")
			currentWord = append(currentWord, subword)
		} else {
			// Start new word
			if len(currentWord) > 0 {
				words = append(words, strings.Join(currentWord, ""))
			}
			currentWord = []string{token}
		}
	}

	// Add final word
	if len(currentWord) > 0 {
		words = append(words, strings.Join(currentWord, ""))
	}

	return strings.Join(words, " "), nil
}

// wordPieceTokenize tokenizes a single word using WordPiece
func (wp *WordPieceTokenizer) wordPieceTokenize(word string) []string {
	if len(word) > wp.maxInputChars {
		return []string{wp.unkToken}
	}

	// Check if word is in vocabulary
	if _, exists := wp.vocab[word]; exists {
		return []string{word}
	}

	// Try to split the word
	start := 0
	var tokens []string

	for start < len(word) {
		end := len(word)
		curSubstr := ""

		// Find the longest subword starting at start
		for start < end {
			substr := word[start:end]
			if start > 0 {
				substr = "##" + substr
			}

			if _, exists := wp.vocab[substr]; exists {
				curSubstr = substr
				break
			}
			end--
		}

		if curSubstr == "" {
			// No subword found, use UNK
			return []string{wp.unkToken}
		}

		tokens = append(tokens, curSubstr)
		start = end
	}

	return tokens
}

// initCharacterVocab initializes vocabulary with characters
func (wp *WordPieceTokenizer) initCharacterVocab(corpus []string) {
	seen := make(map[string]bool)

	// Add special tokens
	wp.addSpecialTokens()

	// Add characters from corpus
	for _, text := range corpus {
		for _, r := range text {
			char := string(r)
			if !seen[char] && !unicode.IsSpace(r) {
				wp.addToken(char)
				seen[char] = true
			}
		}
	}
}

// countWords counts word frequencies in the corpus
func (wp *WordPieceTokenizer) countWords(corpus []string) map[string]int {
	wordCounts := make(map[string]int)

	for _, text := range corpus {
		words := strings.Fields(text)
		for _, word := range words {
			wordCounts[word]++
		}
	}

	return wordCounts
}

// initVocab initializes the vocabulary
func (wp *WordPieceTokenizer) initVocab() {
	// Vocabulary is already initialized with characters and special tokens
}

// findBestSplits finds the best splits to add to vocabulary
func (wp *WordPieceTokenizer) findBestSplits(wordCounts map[string]int) []string {
	type splitScore struct {
		split string
		score float64
	}

	var scores []splitScore

	// For each word, find the best split
	for word, count := range wordCounts {
		if len(word) <= 1 {
			continue
		}

		// Try all possible splits
		for i := 1; i < len(word); i++ {
			left := word[:i]
			right := word[i:]

			// Check if both parts are in vocabulary
			leftInVocab := wp.isInVocab(left)
			rightInVocab := wp.isInVocab(right)

			if leftInVocab && rightInVocab {
				// Calculate score based on frequency and length
				score := float64(count) * math.Log(float64(len(word)))
				// Create subword token for the right part
				subwordToken := "##" + right
				scores = append(scores, splitScore{
					split: subwordToken,
					score: score,
				})
			}
		}
	}

	// Sort by score (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return unique splits
	seen := make(map[string]bool)
	var result []string
	for _, score := range scores {
		if !seen[score.split] {
			result = append(result, score.split)
			seen[score.split] = true
		}
	}

	return result
}

// updateWordCounts updates word counts after adding a new split
func (wp *WordPieceTokenizer) updateWordCounts(wordCounts map[string]int, newSplit string) map[string]int {
	// Remove the ## prefix to get the actual subword
	subword := strings.TrimPrefix(newSplit, "##")

	// Update counts for words that can be split with this subword
	newCounts := make(map[string]int)
	for word, count := range wordCounts {
		if strings.HasSuffix(word, subword) {
			// Split the word and update counts
			prefix := strings.TrimSuffix(word, subword)
			if len(prefix) > 0 {
				newCounts[prefix] += count
				newCounts[newSplit] += count
			} else {
				newCounts[word] = count
			}
		} else {
			newCounts[word] = count
		}
	}

	return newCounts
}

// isInVocab checks if a token is in the vocabulary
func (wp *WordPieceTokenizer) isInVocab(token string) bool {
	_, exists := wp.vocab[token]
	return exists
}

// addToken adds a token to the vocabulary
func (wp *WordPieceTokenizer) addToken(token string) {
	if _, exists := wp.vocab[token]; !exists {
		tokenID := len(wp.vocab)
		wp.vocab[token] = tokenID
		wp.reverseVocab[tokenID] = token
	}
}

// addSpecialTokens adds special tokens to the vocabulary
func (wp *WordPieceTokenizer) addSpecialTokens() {
	tokens := []string{
		wp.specialTokens.PaddingToken,
		wp.specialTokens.EndOfSequence,
		wp.specialTokens.BeginningOfSeq,
		wp.specialTokens.UnknownToken,
		wp.specialTokens.MaskToken,
	}

	for _, token := range tokens {
		wp.addToken(token)
	}

	// Set special token IDs
	wp.paddingTokenID = wp.vocab[wp.specialTokens.PaddingToken]
	wp.eosTokenID = wp.vocab[wp.specialTokens.EndOfSequence]
	wp.bosTokenID = wp.vocab[wp.specialTokens.BeginningOfSeq]
	wp.unknownTokenID = wp.vocab[wp.specialTokens.UnknownToken]
	wp.maskTokenID = wp.vocab[wp.specialTokens.MaskToken]
}

// isSpecialToken checks if a token is special
func (wp *WordPieceTokenizer) isSpecialToken(token string) bool {
	return token == wp.specialTokens.PaddingToken ||
		token == wp.specialTokens.EndOfSequence ||
		token == wp.specialTokens.BeginningOfSeq ||
		token == wp.specialTokens.UnknownToken ||
		token == wp.specialTokens.MaskToken
}

// GetVocabSize returns the vocabulary size
func (wp *WordPieceTokenizer) GetVocabSize() int {
	return wp.vocabSize
}

// GetSpecialTokens returns special tokens
func (wp *WordPieceTokenizer) GetSpecialTokens() SpecialTokens {
	return wp.specialTokens
}

// Save saves the tokenizer to files
func (wp *WordPieceTokenizer) Save(vocabPath, mergePath string) error {
	// Save vocabulary
	vocabData := map[string]interface{}{
		"vocab":          wp.vocab,
		"special_tokens": wp.specialTokens,
	}

	vocabJSON, err := json.MarshalIndent(vocabData, "", "  ")
	if err != nil {
		return err
	}

	err = os.WriteFile(vocabPath, vocabJSON, 0644)
	if err != nil {
		return err
	}

	// WordPiece doesn't use merges, so create empty file
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
func (wp *WordPieceTokenizer) Load(vocabPath, mergePath string) error {
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
				wp.vocab[token] = int(idFloat)
				wp.reverseVocab[int(idFloat)] = token
			}
		}
	}

	// Parse special tokens
	if specialTokensMap, ok := vocabFile["special_tokens"].(map[string]interface{}); ok {
		wp.specialTokens = SpecialTokens{
			PaddingToken:   getString(specialTokensMap, "PaddingToken", ""),
			EndOfSequence:  getString(specialTokensMap, "EndOfSequence", ""),
			BeginningOfSeq: getString(specialTokensMap, "BeginningOfSeq", ""),
			UnknownToken:   getString(specialTokensMap, "UnknownToken", ""),
			MaskToken:      getString(specialTokensMap, "MaskToken", ""),
		}
	}

	wp.vocabSize = len(wp.vocab)
	return nil
}
