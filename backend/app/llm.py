import requests
import json

def get_disease_info(disease_name):
    # Configure your Groq API call
    api_url = "https://api.groq.com/openai/v1/chat/completions"  # Example endpoint
    headers = {
        "Authorization": f"Bearer {YOUR_GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a prompt that requests specific information
    prompt = f"""
    Provide the following information about the plant disease '{disease_name}':
    1. Description: What is this disease and what causes it?
    2. Symptoms: What visual symptoms appear on the plant?
    3. Disease cycle: How does the disease progress?
    4. Treatment options: What are effective treatments?
    5. Prevention methods: How can farmers prevent this disease?
    
    Format the response as JSON with these sections as keys.
    """
    
    # Configure the request payload
    payload = {
        "model": "groq-model-name",  # Replace with actual Groq model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Lower temperature for factual responses
        "max_tokens": 800
    }
    
    # Make the API call
    response = requests.post(api_url, headers=headers, json=payload)
    
    # Process and return the response
    if response.status_code == 200:
        result = response.json()
        response_content = result["choices"][0]["message"]["content"]
        # Parse the JSON response if needed
        try:
            return json.loads(response_content)
        except:
            return response_content
    else:
        return f"Error: {response.status_code}, {response.text}"

def 