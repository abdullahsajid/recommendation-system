from flask import Flask,request,jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
from flask_cors import CORS
from flask_cors import cross_origin
import json

app = Flask(__name__)
CORS(app)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'candidate'
mysql = MySQL(app)
ps = PorterStemmer()



# @app.route('/getdata')
@app.route('/recommend',methods=['GET'])
@cross_origin()
# def getdata():
#     cur = mysql.connect.cursor(MySQLdb.cursors.DictCursor)
#     cur.execute("""SELECT JSON_ARRAYAGG(
#                 JSON_OBJECT(
#                     'id', cp.post_id,
#                     'content', cp.content,
#                     'createdAt', cp.createdAt,
#                     'postImg', cp.url,
#                     'userId', cp.userPost_id,
#                     'comments', (SELECT JSON_ARRAYAGG(
#                                     JSON_OBJECT(
#                                         'id', c.comment_id,
#                                         'comment', c.comment,
#                                         'postId', c.IdPost,
#                                         'userId', c.userComm_id,
#                                         'createdAt', c.createdAt
#                                     )
#                                 )
#                                 FROM comments c 
#                                 WHERE c.IdPost = cp.post_id
#                             ),
#                     'likes', (SELECT JSON_ARRAYAGG(
#                                 JSON_OBJECT(
#                                     'id', candLike.id,
#                                     'PostId', candLike.userPost_id,
#                                     'userId', candLike.usercandId
#                                 )    
#                             )
#                             FROM candidate_post_likes candLike
#                             WHERE candLike.userPost_id = cp.post_id
#                         )      
#                 )
#             )
#             FROM candidatepost cp""")
#     data = cur.fetchall()
#     return jsonify(data)

def home():
    skills = request.args.getlist('skills')
    cur = mysql.connect.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT * FROM candidatepost')

    data = cur.fetchall()
    print("data => ",jsonify(data))
    df_post = pd.DataFrame(data)
    df_post['tags'] = df_post['content']
    df_post['tags'] = df_post['tags'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
    # print("posts",df_post['tags'])
    cv = CountVectorizer(max_features=13000, stop_words="english")
    vector_posts = cv.fit_transform(df_post['tags']).toarray()
    # similar_posts = cosine_similarity(vector_posts)
    def recommend(skills, n=7):
        stemmed_skills = ' '.join([ps.stem(skill) for skill in skills])
        user_vector = cv.transform([stemmed_skills]).toarray()
        similarity_posts = cosine_similarity(user_vector, vector_posts)
        indices = similarity_posts.argsort()[0][::-1][:n]
        recommended_posts = df_post.iloc[indices].to_dict(orient='records')
        return recommended_posts

    # skills = ['javascript','git']
    recommended_posts = recommend(skills)
    cur.close()

    return jsonify(recommended_posts)

if __name__ == "__main__":
    app.run(debug=True)


# cur.execute('SELECT * FROM profile WHERE user_id = 1')
    # profile_data = cur.fetchall()
    # df_profile = pd.DataFrame(profile_data)
    # df_test = df_profile['bio'] + df_profile['experience'] + df_profile['about']
    # print("profile",df_test)



