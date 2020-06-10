from bs4 import BeautifulSoup
import urllib.request
import urllib.parse

fp = open("example1.html")
soup = BeautifulSoup(fp, 'html.parser')

first_div = soup.find("div")
print(first_div)
