library(shiny)
library(reticulate)
library(tidyverse)
# library(tm)
# library(topicmodels)
#library(dplyr)

# Load the data from the fixed CSV file
recode_data=read.csv("updated_recode_project_keywords_20240926_gensim2.csv", stringsAsFactors = F)


# Split keywords into separate rows
knowledge_base <- recode_data %>%
  separate_rows(keywords..LlaMA3.8B., sep = "\n") %>%
  mutate(keywords = trimws(keywords..LlaMA3.8B.))  # Remove any leading/trailing whitespace

# # Create a Document-Term Matrix
# dtm <- DocumentTermMatrix(Corpus(VectorSource(knowledge_base$keywords)))
# 
# # Fit a Latent Dirichlet Allocation (LDA) model
# lda_model <- LDA(dtm, k = 5)  # Adjust 'k' for the number of topics

py_code <- "
import sys
import subprocess

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f'{package} is not installed. Installing...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    finally:
        globals()[package] = __import__(package)


install_and_import('torch')
install_and_import('transformers')
install_and_import('numpy')


from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np


# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to get model response
def get_model_response(user_question):
    inputs = tokenizer.encode_plus(user_question, return_tensors='pt')
    outputs = model(**inputs)
    response = outputs.logits.argmax(dim=-1).item()
    return response
    
# Function to get embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings
"
py_run_string(py_code)
# transformers <- import("transformers")
# tokenizer <- transformers$DistilBertTokenizer$from_pretrained("distilbert-base-uncased")
# model <- transformers$DistilBertForSequenceClassification$from_pretrained("distilbert-base-uncased")

# Compute embeddings for keywords
knowledge_base$embeddings <- lapply(knowledge_base$keywords, function(keyword) {
  py$get_embeddings(keyword)
})

# Define UI
ui <- fluidPage(
  titlePanel("Local Chatbot with Topic Modeling"),
  sidebarLayout(
    sidebarPanel(
      textInput("user_input", "Describe your research and method briefly:", ""),
      actionButton("submit", "Submit")
    ),
    mainPanel(
      textOutput("response"),
      tableOutput("topics")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  observeEvent(input$submit, {
    user_question <- input$user_input
    
    # Get embeddings for the user question
    user_embeddings <- py$get_embeddings(user_question)
    
    # Compute cosine similarity between user question and keywords
    cosine_similarity <- function(a, b) {
      sum(a * b) / (sqrt(sum(a * a)) * sqrt(sum(b * b)))
    }
    
    knowledge_base$similarity <- sapply(knowledge_base$embeddings, function(embedding) {
      cosine_similarity(user_embeddings, embedding)
    })
    
    # Find the most similar projects
    matched_projects <- knowledge_base %>%
      filter(similarity > 0.7) %>%  # Adjust the threshold as needed
      arrange(desc(similarity)) %>% 
      head(5)
    
    response_text <- if (nrow(matched_projects) > 0) {
      paste("Matched projects:", paste(matched_projects$title, collapse = ", "))
    } else {
      "No matching projects found."
    }
    
    # # Extract topics from the response
    # new_dtm <- DocumentTermMatrix(Corpus(VectorSource(user_question)))
    # topic_distribution <- posterior(lda_model, new_dtm)$topics
    
    output$response <- renderText({
      response_text
    })
    
    # output$topics <- renderTable({
    #   as.data.frame(topic_distribution)
    # })
  })
  
  # Trigger initial rendering
  observe({
    updateTextInput(session, "user_input", value = " ")
  
  })
}



# Run the application 
shinyApp(ui = ui, server = server)

# to deploy the shinyApp on GitHub
# organize code in myapp subdir (rename this script to app.R) and run the following command in console
# shinylive::export(appdir = "myapp", destdir = "docs")
# upload to GitHub repo and set page to the docs folder



