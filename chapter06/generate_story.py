# 必要なライブラリのインポート
from openai import AzureOpenAI

# Azure OpenAI Serviceの設定
aoai_endpoint = "https://{Azure OpenAI Serviceのリソース名}.openai.azure.com/"
api_key = "{APIキー}"
api_version = "{APIバージョン}"
deployment_name = "{デプロイ名}"

# Azure OpenAI Serviceのクライアントの作成
openai_client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=aoai_endpoint,
    api_key=api_key
)

# 小説の生成関数
def generate_story(prompt, max_tokens=500):
    response = openai_client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "あなたは小説の作者です。与えられたプロンプトに基づいて、小説を作ってください。"},
            {"role": "user", "content": prompt}
        ],
    )
    story = response.choices[0].message.content
    return story


# プロンプトの設定
prompt = "ワクワクするような楽しいSF小説を作って。"

# 小説の生成
story = generate_story(prompt)

# 生成された小説の表示
print(story)