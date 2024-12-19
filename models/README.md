## Models Folder

This folder is dedicated to storing the necessary model files and weights for the project. You should place the pre-trained model files and weights here to be used by the text generation script.

### Required Model Files

To use the text generation script, you need to have the following files available in this folder:

1. **Pre-trained GPT-2 model**:  
   The GPT-2 model file in PyTorch format (e.g., `pytorch_model.bin`) should be placed here.

2. **Tokenizer files**:  
   The tokenizer files needed for encoding and decoding should be placed here. These typically include:
   - `encoder.json`
   - `vocab.bpe`

### Example Folder Structure

The folder should look like this:

```
/models/
  ├── pytorch_model.bin
  ├── encoder.json
  └── vocab.bpe
```

### How to Download the Model Files

If you don't have the model files, you can download the pre-trained GPT-2 model from [Hugging Face](https://huggingface.co/) or other sources that provide pre-trained GPT-2 models.

Alternatively, you can use the provided `download_file_if_missing` function in the `run.py` script, which will automatically download the model file if it is not found locally.

### Notes

- Ensure that the model files and weights are correctly placed in this directory.
- The `run.py` script will use these model files for text generation. If the model files are missing, it will try to download them automatically.
  
If you encounter issues with the model loading, verify that the files are in the correct format and check for any discrepancies in the file names.

### License

The model files are typically licensed under the terms specified by the model provider (e.g., Hugging Face, OpenAI). Please review the respective license terms when downloading and using the model files.

