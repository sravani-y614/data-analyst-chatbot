import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------
# 1. Initialize OpenRouter LLM
# ---------------------------------------------------------
llm = ChatOpenAI(
    model="qwen/qwen-2.5-72b-instruct",
    openai_api_key="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",  # Replace with your key
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

# Global storage
df_global = None
agent_global = None

# ---------------------------------------------------------
# 2. Handle dataset upload
# ---------------------------------------------------------
def upload_file(file):
    global df_global, agent_global

    if file is None:
        return "⚠️ Please upload a dataset."

    try:
        df_global = pd.read_csv(file.name)

        agent_global = create_pandas_dataframe_agent(
            llm,
            df_global,
            verbose=True,
            allow_dangerous_code=True
        )

        return f"✅ Dataset loaded successfully!\nRows: {len(df_global)} | Columns: {len(df_global.columns)}"

    except Exception as e:
        return f"❌ Error loading dataset: {e}"

# ---------------------------------------------------------
# 3. Visualization Functions
# ---------------------------------------------------------
def plot_order_counts_by_company():
    counts = df_global['company_name'].value_counts()
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=['#4CAF50', '#2196F3']
    ))
    fig.update_layout(title="Order Counts by Company", xaxis_title="Company", yaxis_title="Orders", width=700, height=500)
    return fig

def plot_category_distribution():
    counts = df_global['category'].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.4,
        textinfo="label+percent",
        marker=dict(colors=['#4CAF50','#2196F3','#FFC107','#FF5722','#9C27B0'])
    ))
    fig.update_layout(title="Category Distribution", width=700, height=500)
    return fig

def plot_payment_success_vs_failure():
    summary = df_global.groupby(['company_name','payment_success']).size().unstack(fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary.index, y=summary[True], name='Success', marker_color='#4CAF50'))
    fig.add_trace(go.Bar(x=summary.index, y=summary[False], name='Failed', marker_color='#F44336'))
    fig.update_layout(barmode='stack', title="Payment Success vs Failure by Company", xaxis_title="Company", yaxis_title="Count", width=700, height=500)
    return fig

def plot_avg_price_per_category():
    avg_price = df_global.groupby('category')['price'].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=avg_price.values,
        y=avg_price.index,
        orientation='h',
        marker_color='#2196F3'
    ))
    fig.update_layout(title="Average Price per Category", xaxis_title="Average Price", yaxis_title="Category", width=700, height=500)
    return fig

def plot_heatmap():
    corr = df_global[['price','quantity']].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis'
    ))
    fig.update_layout(title="Correlation Heatmap", width=700, height=500)
    return fig

def plot_price_vs_quantity():
    fig = go.Figure(go.Scatter(
        x=df_global['price'],
        y=df_global['quantity'],
        mode='markers',
        marker=dict(size=8,color='#9C27B0',opacity=0.6)
    ))
    fig.update_layout(title="Price vs Quantity", xaxis_title="Price", yaxis_title="Quantity", width=700, height=500)
    return fig

# ---------------------------------------------------------
# 4. Handle user messages + bot responses
# ---------------------------------------------------------
def add_user_message(message, history):
    if history is None:
        history = []
    history.append({"role": "user", "content": message})
    return "", history

def bot_response(history, chart_type):
    global agent_global, df_global

    if history is None:
        history = []

    if agent_global is None:
        history.append({"role": "assistant", "content": "⚠️ Please upload a dataset first."})
        return history, None

    user_msg = history[-1]["content"]
    fig = None

    try:
        # ✅ Updated to use invoke instead of run
        result = agent_global.invoke({"input": user_msg})["output"]

        # Chart selection logic
        if chart_type == "Horizontal Bar":
            fig = plot_order_counts_by_company()
        elif chart_type == "Donut Chart":
            fig = plot_category_distribution()
        elif chart_type == "Scatter Plot":
            fig = plot_price_vs_quantity()
        elif chart_type == "Stacked Bar":
            fig = plot_payment_success_vs_failure()
        elif chart_type == "Heatmap":
            fig = plot_heatmap()

    except Exception as e:
        result = f"⚠️ Error analyzing data: {e}"

    history.append({"role": "assistant", "content": result})
    return history, fig

# ---------------------------------------------------------
# 5. Build Gradio UI
# ---------------------------------------------------------
with gr.Blocks(theme="soft") as ui:
    gr.Markdown("""
    # 📊 Data Analyst Chatbot  
    Upload a CSV file and choose chart type:
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="📁 Upload CSV Dataset", file_types=[".csv"])
            upload_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(type="messages", height=450)
            plot_output = gr.Plot(label="📈 Visualization")

    msg = gr.Textbox(label="💬 Ask a question about your data", placeholder="e.g., Summarize this dataset")
    chart_dropdown = gr.Dropdown(choices=["Horizontal Bar", "Donut Chart", "Scatter Plot", "Stacked Bar", "Heatmap"], label="Select Chart Type")
    clear_btn = gr.ClearButton([msg, chatbot, plot_output])

    file_upload.upload(upload_file, file_upload, upload_status)
    msg.submit(add_user_message, [msg, chatbot], [msg, chatbot]).then(
        bot_response, [chatbot, chart_dropdown], [chatbot, plot_output]
    )

ui.launch()
