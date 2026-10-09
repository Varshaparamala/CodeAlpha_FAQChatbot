import streamlit as st

from chatbot import FAQChatbot

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 ShopEasy FAQ Chatbot")
st.caption("Ask me about orders, delivery, returns, payments and more.")


@st.cache_resource
def load_bot():
    return FAQChatbot("faqs.csv")


bot = load_bot()

# Sidebar with sample questions
with st.sidebar:
    st.header("Try asking")
    samples = [
        "How do I track my order?",
        "What is your return policy?",
        "How long does a refund take?",
        "Do you accept UPI?",
        "How can I contact support?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True):
            st.session_state["pending"] = s
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! 👋 How can I help you today?"}
    ]

# Show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("matched"):
            st.caption(f"Matched FAQ: {msg['matched']}  |  Confidence: {msg['score']:.0%}")

# Input from the chat box or from a sidebar sample question
user_input = st.chat_input("Type your question here...")
if "pending" in st.session_state:
    user_input = st.session_state.pop("pending")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    result = bot.get_answer(user_input)
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "matched": result["matched"],
            "score": result["score"],
        }
    )
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if result["matched"]:
            st.caption(f"Matched FAQ: {result['matched']}  |  Confidence: {result['score']:.0%}")
