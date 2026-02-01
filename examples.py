"""
Example usage of the enhanced RAG API.
Demonstrates document upload, chunking, and querying.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def example_1_check_api_info():
    """Example 1: Check API information."""
    print("\n" + "="*60)
    print("Example 1: Get API Information")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    info = response.json()
    
    print(f"\nAPI Name: {info['name']}")
    print(f"Version: {info['version']}")
    print(f"Description: {info['description']}")
    print(f"\nSupported Formats: {', '.join(info['supported_formats'])}")
    print(f"\nAvailable Endpoints:")
    for endpoint, description in info['endpoints'].items():
        print(f"  {endpoint}: {description}")


def example_2_add_text_with_chunking():
    """Example 2: Add text with automatic chunking."""
    print("\n" + "="*60)
    print("Example 2: Add Text with Chunking")
    print("="*60)
    
    long_text = """
    Python is a high-level, interpreted programming language known for its simplicity 
    and readability. Created by Guido van Rossum and first released in 1991, Python 
    has become one of the most popular programming languages in the world.
    
    Python supports multiple programming paradigms, including procedural, object-oriented, 
    and functional programming. Its design philosophy emphasizes code readability with 
    the use of significant indentation.
    
    Key features of Python include:
    - Dynamic typing and automatic memory management
    - Comprehensive standard library
    - Support for modules and packages
    - Exception handling
    - High-level data structures
    
    Python is widely used in web development, data science, artificial intelligence, 
    scientific computing, automation, and many other domains. Popular frameworks include 
    Django and Flask for web development, NumPy and Pandas for data analysis, and 
    TensorFlow and PyTorch for machine learning.
    """
    
    response = requests.post(
        f"{BASE_URL}/add",
        params={
            "text": long_text,
            "chunk": True,
            "strategy": "recursive"
        }
    )
    
    result = response.json()
    print(f"\nStatus: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Number of chunks created: {result['chunks']}")
    print(f"First chunk ID: {result['ids'][0]}")


def example_3_upload_markdown_file():
    """Example 3: Upload a markdown file."""
    print("\n" + "="*60)
    print("Example 3: Upload Markdown File")
    print("="*60)
    
    # Create a sample markdown file
    markdown_content = """# Data Structures in Computer Science

## Arrays
Arrays are contiguous blocks of memory that store elements of the same type. They provide 
O(1) access time but have fixed size in most languages.

## Linked Lists
Linked lists consist of nodes where each node contains data and a reference to the next node. 
They allow dynamic size but have O(n) access time.

## Trees
Trees are hierarchical data structures with a root node and child nodes. Binary trees, AVL trees, 
and B-trees are common variants used in databases and file systems.

## Hash Tables
Hash tables use a hash function to map keys to values, providing average O(1) insertion and 
lookup time. They are widely used for implementing dictionaries and sets.

## Graphs
Graphs consist of vertices and edges, representing relationships between entities. They are 
used in social networks, routing algorithms, and recommendation systems.
"""
    
    import tempfile
    import os
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(markdown_content)
        temp_file = f.name
    
    try:
        with open(temp_file, 'rb') as f:
            files = {'file': ('data_structures.md', f, 'text/markdown')}
            data = {
                'strategy': 'semantic',
                'chunk_size': 500,
                'chunk_overlap': 100
            }
            
            response = requests.post(f"{BASE_URL}/upload", files=files, data=data)
            result = response.json()
            
            print(f"\nStatus: {result['status']}")
            print(f"Filename: {result['filename']}")
            print(f"File Type: {result['file_type']}")
            print(f"Number of chunks: {result['chunks']}")
            print(f"\nMetadata:")
            for key, value in result['metadata'].items():
                if key not in ['page_texts', 'sections']:  # Skip large nested data
                    print(f"  {key}: {value}")
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def example_4_query_with_sources():
    """Example 4: Query the knowledge base and get sources."""
    print("\n" + "="*60)
    print("Example 4: Query with Source Metadata")
    print("="*60)
    
    questions = [
        "What is Python used for?",
        "What are hash tables?",
        "Explain linked lists"
    ]
    
    for question in questions:
        print(f"\n📝 Question: {question}")
        
        response = requests.post(
            f"{BASE_URL}/query",
            params={"q": question}
        )
        
        result = response.json()
        
        print(f"\n💡 Answer: {result['answer']}\n")
        
        if result.get('sources'):
            print(f"📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  Source {i}:")
                print(f"    Preview: {source['text_preview'][:80]}...")
                
                metadata = source.get('metadata', {})
                if metadata:
                    print(f"    Metadata:")
                    for key, value in metadata.items():
                        if key not in ['page_texts', 'sections']:
                            print(f"      - {key}: {value}")
        
        print("\n" + "-"*60)


def example_5_different_chunking_strategies():
    """Example 5: Compare different chunking strategies."""
    print("\n" + "="*60)
    print("Example 5: Different Chunking Strategies")
    print("="*60)
    
    sample_text = """
    Artificial Intelligence (AI) is transforming industries worldwide. Machine learning, 
    a subset of AI, enables computers to learn from data without explicit programming. 
    Deep learning, using neural networks, has achieved breakthroughs in image recognition 
    and natural language processing. AI applications include autonomous vehicles, medical 
    diagnosis, fraud detection, and personalized recommendations.
    """
    
    strategies = ['recursive', 'semantic']
    
    for strategy in strategies:
        print(f"\n--- Using {strategy.upper()} strategy ---")
        
        response = requests.post(
            f"{BASE_URL}/add",
            params={
                "text": sample_text,
                "chunk": True,
                "strategy": strategy
            }
        )
        
        result = response.json()
        print(f"Chunks created: {result['chunks']}")
        print(f"Strategy: {strategy}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*60)
    print("RAG API - Usage Examples")
    print("="*60)
    
    examples = [
        example_1_check_api_info,
        example_2_add_text_with_chunking,
        example_3_upload_markdown_file,
        example_4_query_with_sources,
        example_5_different_chunking_strategies
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {str(e)}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    run_all_examples()
