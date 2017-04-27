curl -H "Content-Type: application/json" -X POST -d '{"username":"xyz","password":"xyz"}' http://localhost:9000/api/imagenet/predict


curl -H "Content-Type: application/json" \
    -X POST -d '{"image_url": "http://www.hawkshop.com/webitemimages/103/CW1702-t.jpg"}' \
    http://localhost:9000/api/imagenet/predict
