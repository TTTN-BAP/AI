from flask import Flask, request, make_response, jsonify
import json
from ContentBased_Recommender import ContentBased


app = Flask(__name__)

@app.after_request 
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = '*'
    header['Access-Control-Allow-Methods'] = '*'
    # Other headers can be added here if needed
    return response

content_based_model = ContentBased()

@app.route('/content-based-with-cv', methods = ['POST'])
def recommend_content_based_with_cv():
    data = request.get_json()
    cv_target = data['cv_target']
    cv_academi_level = data['cv_academi_level']
    cv_skill = data['cv_skill']
    cv_interest = data['cv_interest']
    result = content_based_model.recommend_with_cv(cv_target, cv_academi_level, cv_skill, cv_interest)
    # Chuyển đổi đối tượng JSON sang chuỗi JSON bằng json.dumps()
    result_json = json.dumps(result)
        
    # Tạo response với chuỗi JSON và mã trạng thái HTTP 200
    response = make_response(result_json)
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/content-based-with-job', methods = ['POST'])
def recommend_content_based_with_job():
    data = request.get_json()
    #job_id = request.args.get('id')
    job_id = data['id']
    print(job_id)
    result = content_based_model.recommend_with_job(job_id)
    # Chuyển đổi đối tượng JSON sang chuỗi JSON bằng json.dumps()
    result_json = json.dumps(result)
    # Tạo response với chuỗi JSON và mã trạng thái HTTP 200
    response = make_response(result_json)
    response.headers['Content-Type'] = 'application/json'
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)