# LangChain with Gemini and Streamlit

This project demonstrates how to integrate LangChain with Google Generative AI (Gemini) to build applications that process text, images, and PDFs. The project includes three main functionalities:

1. **Image Processing with Text**
2. **Chat Model**
3. **PDF Reader and Question Answering**

## Features

- **Image Processing**: Upload an image and input text to receive AI-generated responses based on the image and text input.
- **Chat Model**: Input text to interact with an AI chat model.
- **PDF Reader**: Upload PDF files and ask questions about their content.

## Setup

### Prerequisites

- Python 3.8 or higher
- Streamlit
- LangChain
- Google Generative AI API Key

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/alihassanml/langchain-gemini-streamlit.git
   cd langchain-gemini-streamlit
   ```

2. **Create a Virtual Environment and Activate It**

   ```sh
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install Required Packages**

   ```sh
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your Google Generative AI API key:

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

Run the Streamlit application:

```sh
streamlit run app.py
```

### Navigating the Application

- **Sidebar**: Use the sidebar to navigate between different functionalities.
  - `Image Processing`: Upload an image and input text to receive responses based on the image and text.
  - `Langchain Chat Model`: Input text to interact with the chat model.
  - `Langchain PDF`: Upload PDF files and ask questions about their content.

### Image Processing

1. Navigate to `Image Processing`.
2. Enter text in the input field.
3. Upload an image (JPG, PNG, JPEG).
4. Click `Ask Question`.
5. View the response generated based on the text and image.

### Chat Model

1. Navigate to `Langchain Chat Model`.
2. Enter text in the input field.
3. Click `Ask Question`.
4. View the response from the chat model.

### PDF Reader

1. Navigate to `Langchain PDF`.
2. Upload PDF files using the sidebar.
3. Click `Submit File` to process the PDFs.
4. Enter a question in the text input field.
5. Click `Ask Question` to get a response based on the content of the uploaded PDFs.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Developed by Ali Hassan

- LinkedIn: [Connect LinkedIn](https://www.linkedin.com/in/alihassanml)
- GitHub: [Connect On GitHub](https://github.com/alihassanml)
