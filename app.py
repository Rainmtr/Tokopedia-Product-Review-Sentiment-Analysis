from flask import Flask, render_template, request
import requests
import json
import urllib.parse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)

# Define API endpoints and headers
getProductIdAPI = 'https://gql.tokopedia.com/graphql/PDPGetLayoutQuery'
getProductIdHeader = {
    'sec-ch-ua': '',
    'X-Version': '',
    'X-TKPD-AKAMAI': 'pdpGetLayout',
    'sec-ch-ua-mobile': '',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
    'content-type': 'application/json',
    'accept': '*/*',
    'Referer': '',
    'X-Source': '',
    'x-device': '',
    'X-Tkpd-Lite-Service': '',
    'sec-ch-ua-platform': ''
}

getReviewAPI = 'https://gql.tokopedia.com/graphql/productReviewList'
getReviewHeader = {
    'sec-ch-ua': '',
    'X-Version': '',
    'sec-ch-ua-mobile': '',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
    'content-type': 'application/json',
    'accept': '*/*',
    'Referer': '',
    'X-Source': '',
    'X-Tkpd-Lite-Service': '',
    'sec-ch-ua-platform': ''
}

def roundUp(number):
    return int(number / 50) + (number % 50 > 0)

def extract_shopDomain_productKey(url):
    parsed_url = urllib.parse.urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) >= 2:
        shopDomain = path_parts[0]
        productKey = path_parts[1]
        return shopDomain, productKey
    else:
        raise ValueError("Invalid Tokopedia product URL format")

def getReviewData(pages, product_id):
    return [
        {
            "operationName": "productReviewList",
            "variables": {
                "productID": product_id,
                "page": pages,
                "limit": 50,
                "sortBy": "create_time desc",
                "filterBy": ""
            },
            "query": "query productReviewList($productID: String!, $page: Int!, $limit: Int!, $sortBy: String, $filterBy: String) {\n  productrevGetProductReviewList(productID: $productID, page: $page, limit: $limit, sortBy: $sortBy, filterBy: $filterBy) {\n    productID\n    list {\n      id: feedbackID\n      message\n      }\n    }\n}\n"
        }
    ]

def getReviews(pages, product_id):
    reviewList = []
    for i in range(pages):
        reviewData = getReviewData(i + 1, product_id)
        reviewResponse = requests.post(getReviewAPI, headers=getReviewHeader, data=json.dumps(reviewData))
        reviewResponseData = reviewResponse.json()
        reviews = reviewResponseData[0]['data']['productrevGetProductReviewList']['list']
        reviewList.extend(reviews)
    messages = [review['message'].replace('\n', ' ').strip() for review in reviewList]
    return messages

def analyze_sentiment(reviews):
    fined_model_path = 'C:/Users/adria/product_sentiment_analysis/fined_model'
    tokenizer_path = 'C:/Users/adria/product_sentiment_analysis/fined_tokenizer'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(fined_model_path)

    sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    results = sentiment_analysis(reviews)

    labels = [result['label'] for result in results]
    scores = [result['score'] for result in results]

    reverse_label_mapping = {'LABEL_0': -1, 'LABEL_1': 0, 'LABEL_2': 1}
    mapped_labels = [reverse_label_mapping[label] for label in labels]

    total_reviews = len(mapped_labels)
    positive_reviews = mapped_labels.count(1)
    neutral_reviews = mapped_labels.count(0)
    negative_reviews = mapped_labels.count(-1)

    positive_percentage = (positive_reviews / total_reviews) * 100
    neutral_percentage = (neutral_reviews / total_reviews) * 100
    negative_percentage = (negative_reviews / total_reviews) * 100

    return {
        "total_reviews": total_reviews,
        "positive_reviews": positive_reviews,
        "negative_reviews": negative_reviews,
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage
    }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process_link', methods=['POST'])
def process_link():
    tokopedia_link = request.form['tokopedia_link']
    shopDomain, productKey = extract_shopDomain_productKey(tokopedia_link)

    # Retrieve Product ID
    getProductIdData = [
        {
            "operationName": "PDPGetLayoutQuery",
            "variables": {
                "shopDomain": shopDomain,
                "productKey": productKey,
                "layoutID": "",
                "apiVersion": 1,
                "extParam": ""
            },
            "query": "query PDPGetLayoutQuery($shopDomain: String, $productKey: String, $layoutID: String, $apiVersion: Float, $userLocation: pdpUserLocation, $extParam: String, $tokonow: pdpTokoNow, $deviceID: String) {\n  pdpGetLayout(shopDomain: $shopDomain, productKey: $productKey, layoutID: $layoutID, apiVersion: $apiVersion, userLocation: $userLocation, extParam: $extParam, tokonow: $tokonow, deviceID: $deviceID) {\n    basicInfo {\n      id: productID\n      shopID\n      }\n    }\n}\n"
        }
    ]

    getProductIdResponse = requests.post(getProductIdAPI, headers=getProductIdHeader, data=json.dumps(getProductIdData))
    productIdResponseData = getProductIdResponse.json()
    product_id = productIdResponseData[0]['data']['pdpGetLayout']['basicInfo']['id']

    # Retrieve Reviews
    pages = 1
    getTotalReviewData = [
        {
            "operationName": "productReviewList",
            "variables": {
                "productID": product_id,
                "page": pages,
                "limit": 50,
                "sortBy": "create_time desc",
                "filterBy": ""
            },
            "query": "query productReviewList($productID: String!, $page: Int!, $limit: Int!, $sortBy: String, $filterBy: String) {\n  productrevGetProductReviewList(productID: $productID, page: $page, limit: $limit, sortBy: $sortBy, filterBy: $filterBy) {\n    productID\n    totalReviews\n    }\n}\n"
        }
    ]

    totalReviewResponse = requests.post(getReviewAPI, headers=getReviewHeader, data=json.dumps(getTotalReviewData))
    totalReviewData = totalReviewResponse.json()
    total_reviews = totalReviewData[0]['data']['productrevGetProductReviewList']['totalReviews']

    pages = roundUp(total_reviews)
    reviewList = getReviews(pages, product_id)
    stats = analyze_sentiment(reviewList)

    result = (
        f"Product ID: {product_id}\n"
        f"Total Reviews: {total_reviews}\n"
        f"Positive Reviews: {stats['positive_reviews']} ({stats['positive_percentage']:.2f}%)\n"
        f"Negative Reviews: {stats['negative_reviews']} ({stats['negative_percentage']:.2f}%)\n"
    )

    return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
