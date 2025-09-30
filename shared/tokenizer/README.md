# Tokenizer Package

A comprehensive Go package implementing popular tokenization strategies used in modern language models, including BPE, WordPiece, and SentencePiece.

## Features

- **Byte Pair Encoding (BPE)** - Used by GPT models
- **WordPiece** - Used by BERT models  
- **SentencePiece** - Google's universal tokenizer
- **GPT-Compatible Tokenizer** - Pre-configured for GPT models
- **BERT-Compatible Tokenizer** - Pre-configured for BERT models
- **Save/Load Functionality** - Persistent tokenizers
- **Custom Special Tokens** - Configurable special token handling
- **Comprehensive Testing** - Full test coverage

## Installation

```go
import "github.com/pilillo/yannp/shared/tokenizer"
```

## Quick Start

### Basic Usage

```go
package main

import (
    "fmt"
    "log"
    "github.com/pilillo/yannp/shared/tokenizer"
)

func main() {
    // Create a BPE tokenizer
    tokenizer := tokenizer.NewBPETokenizer(tokenizer.DefaultSpecialTokens())
    
    // Training corpus
    corpus := []string{
        "hello world",
        "hello there", 
        "world peace",
        "hello beautiful world",
        "peace and love",
    }
    
    // Train the tokenizer
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
    
    // Decode back to text
    decoded, err := tokenizer.Decode(tokenIDs)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Decoded: %s\n", decoded)
}
```

### GPT-Compatible Tokenization

```go
// Create GPT tokenizer
gptTokenizer := tokenizer.NewGPTTokenizer()

// Train on your corpus
corpus := []string{
    "Hello, world! How are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating.",
}

err := gptTokenizer.Train(corpus, 1000)
if err != nil {
    log.Fatal(err)
}

// Encode text
text := "Hello, world! How are you today?"
tokenIDs, err := gptTokenizer.Encode(text)
if err != nil {
    log.Fatal(err)
}

// Decode back
decoded, err := gptTokenizer.Decode(tokenIDs)
if err != nil {
    log.Fatal(err)
}
```

### BERT-Compatible Tokenization

```go
// Create BERT tokenizer
bertTokenizer := tokenizer.NewBERTTokenizer()

// Train on your corpus
corpus := []string{
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! How are you today?",
    "Machine learning is fascinating.",
}

err := bertTokenizer.Train(corpus, 1000)
if err != nil {
    log.Fatal(err)
}

// Encode text
text := "The quick brown fox jumps over the lazy dog."
tokenIDs, err := bertTokenizer.Encode(text)
if err != nil {
    log.Fatal(err)
}

// Decode back
decoded, err := bertTokenizer.Decode(tokenIDs)
if err != nil {
    log.Fatal(err)
}
```

## Tokenizer Types

### Byte Pair Encoding (BPE)

BPE is used by GPT models and works by iteratively merging the most frequent character pairs.

```go
bpeTokenizer := tokenizer.NewBPETokenizer(tokenizer.DefaultSpecialTokens())
err := bpeTokenizer.Train(corpus, vocabSize)
```

**Features:**
- Character-level initialization
- Iterative pair merging
- Handles unknown words gracefully
- Compatible with GPT models

### WordPiece

WordPiece is used by BERT models and uses a different merging strategy based on likelihood.

```go
wordpieceTokenizer := tokenizer.NewWordPieceTokenizer(tokenizer.DefaultSpecialTokens())
err := wordpieceTokenizer.Train(corpus, vocabSize)
```

**Features:**
- Subword tokenization with `##` prefix
- Likelihood-based merging
- Compatible with BERT models
- Handles out-of-vocabulary words

### SentencePiece

SentencePiece is Google's universal tokenizer that works directly on raw text.

```go
sentencepieceTokenizer := tokenizer.NewSentencePieceTokenizer(tokenizer.DefaultSpecialTokens())
err := sentencepieceTokenizer.Train(corpus, vocabSize)
```

**Features:**
- Works on raw text (no preprocessing)
- Multiple algorithms (unigram, BPE, char, word)
- Language-agnostic
- Handles multiple languages

## Special Tokens

All tokenizers support special tokens for common use cases:

```go
specialTokens := tokenizer.SpecialTokens{
    PaddingToken:   "<|pad|>",
    EndOfSequence:  "<|endoftext|>",
    BeginningOfSeq: "<|startoftext|>",
    UnknownToken:   "<|unk|>",
    MaskToken:      "<|mask|>",
}

tokenizer := tokenizer.NewBPETokenizer(specialTokens)
```

## Save and Load

Tokenizers can be saved to and loaded from files:

```go
// Save tokenizer
err := tokenizer.Save("vocab.json", "merges.json")
if err != nil {
    log.Fatal(err)
}

// Load tokenizer
newTokenizer := tokenizer.NewBPETokenizer(tokenizer.DefaultSpecialTokens())
err = newTokenizer.Load("vocab.json", "merges.json")
if err != nil {
    log.Fatal(err)
}
```

## Model Compatibility

### GPT Models

Use the GPT tokenizer for GPT-style models:

```go
gptTokenizer := tokenizer.NewGPTTokenizer()
```

**Compatible with:**
- GPT-2
- GPT-3
- GPT-4
- ChatGPT
- Other GPT variants

### BERT Models

Use the BERT tokenizer for BERT-style models:

```go
bertTokenizer := tokenizer.NewBERTTokenizer()
```

**Compatible with:**
- BERT
- RoBERTa
- DistilBERT
- Other BERT variants

## Advanced Usage

### Custom Special Tokens

```go
customTokens := tokenizer.SpecialTokens{
    PaddingToken:   "<pad>",
    EndOfSequence:  "<eos>",
    BeginningOfSeq: "<bos>",
    UnknownToken:   "<unk>",
    MaskToken:      "<mask>",
}

tokenizer := tokenizer.NewBPETokenizer(customTokens)
```

### Tokenizer Comparison

```go
text := "Hello, world! How are you today?"

// Create different tokenizers
bpeTokenizer := tokenizer.NewBPETokenizer(tokenizer.DefaultSpecialTokens())
wordpieceTokenizer := tokenizer.NewWordPieceTokenizer(tokenizer.DefaultSpecialTokens())
sentencepieceTokenizer := tokenizer.NewSentencePieceTokenizer(tokenizer.DefaultSpecialTokens())

// Train all tokenizers
corpus := []string{text}
bpeTokenizer.Train(corpus, 100)
wordpieceTokenizer.Train(corpus, 100)
sentencepieceTokenizer.Train(corpus, 100)

// Compare results
bpeTokens, _ := bpeTokenizer.Encode(text)
wordpieceTokens, _ := wordpieceTokenizer.Encode(text)
sentencepieceTokens, _ := sentencepieceTokenizer.Encode(text)

fmt.Printf("BPE: %v\n", bpeTokens)
fmt.Printf("WordPiece: %v\n", wordpieceTokens)
fmt.Printf("SentencePiece: %v\n", sentencepieceTokens)
```

## Testing

Run the tests:

```bash
go test ./shared/tokenizer/... -v
```

## Examples

See the `examples.go` file for comprehensive usage examples:

- Basic tokenization
- GPT-compatible tokenization
- BERT-compatible tokenization
- Save/load functionality
- Custom special tokens
- Tokenizer comparison

## Performance

The tokenizers are optimized for performance:

- **BPE**: Fast training and encoding
- **WordPiece**: Efficient subword tokenization
- **SentencePiece**: High-speed universal tokenization
- **Memory efficient**: Minimal memory footprint
- **Concurrent safe**: Can be used in goroutines

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Byte Pair Encoding](https://arxiv.org/abs/1508.07909)
- [WordPiece](https://arxiv.org/abs/1609.08144)
- [SentencePiece](https://arxiv.org/abs/1808.06226)
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT](https://arxiv.org/abs/2005.14165)
