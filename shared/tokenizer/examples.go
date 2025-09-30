package tokenizer

import (
	"fmt"
	"log"
)

// ExampleBPE demonstrates BPE tokenization
func ExampleBPE() {
	// Create BPE tokenizer
	tokenizer := NewBPETokenizer(DefaultSpecialTokens())

	// Training corpus
	corpus := []string{
		"hello world",
		"hello there",
		"world peace",
		"hello beautiful world",
		"peace and love",
	}

	// Train tokenizer
	err := tokenizer.Train(corpus, 1000)
	if err != nil {
		log.Fatal(err)
	}

	// Encode text
	text := "hello beautiful world"
	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Token IDs: %v\n", tokenIDs)

	// Decode back
	decoded, err := tokenizer.Decode(tokenIDs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Decoded: %s\n", decoded)
}

// ExampleWordPiece demonstrates WordPiece tokenization
func ExampleWordPiece() {
	// Create WordPiece tokenizer
	tokenizer := NewWordPieceTokenizer(DefaultSpecialTokens())

	// Training corpus
	corpus := []string{
		"the quick brown fox",
		"jumps over the lazy dog",
		"the quick brown fox jumps",
		"over the lazy dog",
		"brown fox jumps over",
	}

	// Train tokenizer
	err := tokenizer.Train(corpus, 1000)
	if err != nil {
		log.Fatal(err)
	}

	// Encode text
	text := "the quick brown fox"
	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Token IDs: %v\n", tokenIDs)

	// Decode back
	decoded, err := tokenizer.Decode(tokenIDs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Decoded: %s\n", decoded)
}

// ExampleSentencePiece demonstrates SentencePiece tokenization
func ExampleSentencePiece() {
	// Create SentencePiece tokenizer
	tokenizer := NewSentencePieceTokenizer(DefaultSpecialTokens())

	// Training corpus
	corpus := []string{
		"Hello, world! How are you today?",
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is fascinating.",
		"Tokenization is an important preprocessing step.",
	}

	// Train tokenizer
	err := tokenizer.Train(corpus, 1000)
	if err != nil {
		log.Fatal(err)
	}

	// Encode text
	text := "Hello, world! How are you today?"
	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Token IDs: %v\n", tokenIDs)

	// Decode back
	decoded, err := tokenizer.Decode(tokenIDs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Decoded: %s\n", decoded)
}

// ExampleGPT demonstrates GPT-compatible tokenization
func ExampleGPT() {
	// Create GPT tokenizer
	tokenizer := NewGPTTokenizer()

	// Training corpus
	corpus := []string{
		"Hello, world! How are you today?",
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is fascinating.",
		"Tokenization is an important preprocessing step.",
	}

	// Train tokenizer
	err := tokenizer.Train(corpus, 1000)
	if err != nil {
		log.Fatal(err)
	}

	// Encode text
	text := "Hello, world! How are you today?"
	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Token IDs: %v\n", tokenIDs)

	// Decode back
	decoded, err := tokenizer.Decode(tokenIDs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Decoded: %s\n", decoded)
}

// ExampleBERT demonstrates BERT-compatible tokenization
func ExampleBERT() {
	// Create BERT tokenizer
	tokenizer := NewBERTTokenizer()

	// Training corpus
	corpus := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Hello, world! How are you today?",
		"Machine learning is fascinating.",
		"Tokenization is an important preprocessing step.",
	}

	// Train tokenizer
	err := tokenizer.Train(corpus, 1000)
	if err != nil {
		log.Fatal(err)
	}

	// Encode text
	text := "The quick brown fox jumps over the lazy dog."
	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Token IDs: %v\n", tokenIDs)

	// Decode back
	decoded, err := tokenizer.Decode(tokenIDs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Decoded: %s\n", decoded)
}

// ExampleSaveLoad demonstrates saving and loading tokenizers
func ExampleSaveLoad() {
	// Create and train tokenizer
	tokenizer := NewBPETokenizer(DefaultSpecialTokens())
	corpus := []string{
		"hello world",
		"hello there",
		"world peace",
		"hello beautiful world",
		"peace and love",
	}

	err := tokenizer.Train(corpus, 1000)
	if err != nil {
		log.Fatal(err)
	}

	// Save tokenizer
	err = tokenizer.Save("vocab.json", "merges.json")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Tokenizer saved successfully")

	// Create new tokenizer and load
	newTokenizer := NewBPETokenizer(DefaultSpecialTokens())
	err = newTokenizer.Load("vocab.json", "merges.json")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Tokenizer loaded successfully")

	// Test loaded tokenizer
	text := "hello beautiful world"
	tokenIDs, err := newTokenizer.Encode(text)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Text: %s\n", text)
	fmt.Printf("Token IDs: %v\n", tokenIDs)
}

// ExampleCustomSpecialTokens demonstrates using custom special tokens
func ExampleCustomSpecialTokens() {
	// Create custom special tokens
	specialTokens := SpecialTokens{
		PaddingToken:   "<pad>",
		EndOfSequence:  "<eos>",
		BeginningOfSeq: "<bos>",
		UnknownToken:   "<unk>",
		MaskToken:      "<mask>",
	}

	// Create tokenizer with custom special tokens
	tokenizer := NewBPETokenizer(specialTokens)

	// Training corpus
	corpus := []string{
		"hello world",
		"hello there",
		"world peace",
	}

	// Train tokenizer
	err := tokenizer.Train(corpus, 100)
	if err != nil {
		log.Fatal(err)
	}

	// Test special tokens
	specialTokensFromTokenizer := tokenizer.GetSpecialTokens()
	fmt.Printf("Padding Token: %s\n", specialTokensFromTokenizer.PaddingToken)
	fmt.Printf("End of Sequence Token: %s\n", specialTokensFromTokenizer.EndOfSequence)
	fmt.Printf("Beginning of Sequence Token: %s\n", specialTokensFromTokenizer.BeginningOfSeq)
	fmt.Printf("Unknown Token: %s\n", specialTokensFromTokenizer.UnknownToken)
	fmt.Printf("Mask Token: %s\n", specialTokensFromTokenizer.MaskToken)
}

// ExampleTokenizerComparison demonstrates comparing different tokenizers
func ExampleTokenizerComparison() {
	text := "Hello, world! How are you today?"

	// Create different tokenizers
	bpeTokenizer := NewBPETokenizer(DefaultSpecialTokens())
	wordpieceTokenizer := NewWordPieceTokenizer(DefaultSpecialTokens())
	sentencepieceTokenizer := NewSentencePieceTokenizer(DefaultSpecialTokens())
	gptTokenizer := NewGPTTokenizer()
	bertTokenizer := NewBERTTokenizer()

	// Training corpus
	corpus := []string{
		"Hello, world! How are you today?",
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is fascinating.",
		"Tokenization is an important preprocessing step.",
	}

	// Train all tokenizers

	// Train each tokenizer individually
	bpeTokenizer.Train(corpus, 100)
	wordpieceTokenizer.Train(corpus, 100)
	sentencepieceTokenizer.Train(corpus, 100)
	gptTokenizer.Train(corpus, 100)
	bertTokenizer.Train(corpus, 100)

	// Test each tokenizer individually
	fmt.Printf("BPE Tokenizer:\n")
	bpeTokens, _ := bpeTokenizer.Encode(text)
	bpeDecoded, _ := bpeTokenizer.Decode(bpeTokens)
	fmt.Printf("  Original: %s\n", text)
	fmt.Printf("  Token IDs: %v\n", bpeTokens)
	fmt.Printf("  Decoded: %s\n", bpeDecoded)
	fmt.Printf("  Vocab Size: %d\n", bpeTokenizer.GetVocabSize())
	fmt.Println()

	fmt.Printf("WordPiece Tokenizer:\n")
	wpTokens, _ := wordpieceTokenizer.Encode(text)
	wpDecoded, _ := wordpieceTokenizer.Decode(wpTokens)
	fmt.Printf("  Original: %s\n", text)
	fmt.Printf("  Token IDs: %v\n", wpTokens)
	fmt.Printf("  Decoded: %s\n", wpDecoded)
	fmt.Printf("  Vocab Size: %d\n", wordpieceTokenizer.GetVocabSize())
	fmt.Println()

	fmt.Printf("SentencePiece Tokenizer:\n")
	spTokens, _ := sentencepieceTokenizer.Encode(text)
	spDecoded, _ := sentencepieceTokenizer.Decode(spTokens)
	fmt.Printf("  Original: %s\n", text)
	fmt.Printf("  Token IDs: %v\n", spTokens)
	fmt.Printf("  Decoded: %s\n", spDecoded)
	fmt.Printf("  Vocab Size: %d\n", sentencepieceTokenizer.GetVocabSize())
	fmt.Println()

	fmt.Printf("GPT Tokenizer:\n")
	gptTokens, _ := gptTokenizer.Encode(text)
	gptDecoded, _ := gptTokenizer.Decode(gptTokens)
	fmt.Printf("  Original: %s\n", text)
	fmt.Printf("  Token IDs: %v\n", gptTokens)
	fmt.Printf("  Decoded: %s\n", gptDecoded)
	fmt.Printf("  Vocab Size: %d\n", gptTokenizer.GetVocabSize())
	fmt.Println()

	fmt.Printf("BERT Tokenizer:\n")
	bertTokens, _ := bertTokenizer.Encode(text)
	bertDecoded, _ := bertTokenizer.Decode(bertTokens)
	fmt.Printf("  Original: %s\n", text)
	fmt.Printf("  Token IDs: %v\n", bertTokens)
	fmt.Printf("  Decoded: %s\n", bertDecoded)
	fmt.Printf("  Vocab Size: %d\n", bertTokenizer.GetVocabSize())
	fmt.Println()
}
