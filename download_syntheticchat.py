import pandas as pd

def preprocess_synthetic_dataset():
    input_file = "C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/synthetic_therapy_conversations.csv"
    output_file = "C:/Users/hp/Downloads/AI_Chatbot_Mental_Health_Support/processed_synthetic_chatbot_data.csv"

    df = pd.read_csv(input_file)

    if 'conversations' not in df.columns:
        raise ValueError("The dataset must have a 'conversations' column.")

    data = []

    for convo in df['conversations'].dropna():   
        lines = convo.split('\n')
        for i in range(len(lines) - 1):
            input_text = lines[i].strip()
            output_text = lines[i + 1].strip()
            if input_text and output_text:
                data.append({'input': input_text, 'output': output_text})

    processed_df = pd.DataFrame(data)
    processed_df.to_csv(output_file, index=False)

    print(f"Processed and saved to: {output_file}")
    print(processed_df.head())

if __name__ == "__main__":
    preprocess_synthetic_dataset()
