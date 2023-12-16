import requests

url = 'http://localhost:5000/upload'
files = {
    'pdf_or_json': open(r'sample.json', 'rb'),
    'json': open(r'questionsforsamplejson.json', 'rb')
}
response = requests.post(url, files=files)
# print(response.text)