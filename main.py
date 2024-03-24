import streamlit as st
import qaretriever as qa

urls = []

if 'i' not in st.session_state:
    st.session_state.i = 1


if 'urls_added' not in st.session_state:
    st.session_state.urls_added = 1
st.title('URL Agent ')
st.write('This is a simple app to retrieve answers from a list of urls')
st.sidebar.title('URLS')
question = ''
question = st.text_input('Ask a question:')
max_urls = 4
loader = st.empty()
def add_url_input():
    if st.session_state.urls_added < max_urls:
        st.session_state.i += 1
        st.session_state.urls_added += 1

for i in range(st.session_state.i):
    url_input = st.sidebar.text_input(f'URL {i + 1}:', key=i)
    urls.append(url_input)

print('urls',urls)
st.sidebar.markdown("---")
if st.sidebar.button('Add URL'):
    add_url_input()
process = st.sidebar.button('Process URLs')
if process:
    print('question is',question,'urls are',urls)
    final = qa.qaretriever(question,urls)
    if final:
        st.header('ANSWER')
        if urls[0].startswith('https://www.youtube.com'):
            st.write(final['result'])
        else:
            st.write(final['result'])

    else:
        st.write('No answer found!')