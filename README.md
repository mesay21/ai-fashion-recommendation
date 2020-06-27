# ai-fashion-recommendation

# DressUp

Leveraging natural language processing and deep learning technology to help English language learners on the road to
fluency.

DressUp is an AI-based fashion recommendation system for complete outfit based on the decision behavior of fashion experts. This repo includes a webapp that demonstrate the structure of the recommendation system. This product is created for a fashion website to give a complete outfit recommendation for the website's users based on the items in the customers' shopping cart or wardrobes. 

## Quick Start

Dockerfile.train creates the training environment with torch 1.5, torchvision 0.6 which outputs the trained weights in the models folder. 
Dockerfile.api creates the environment for the webapp/api that provides the outfit recommendation in real time. 

## What problem does DressUp address?

People have so many clothes that they never wear. For instance, majority of the time, they end up wearing just a few clothes after spending tons of hours in front their wardrobes. One of the biggest reasons is that most of the clothes they buy are not compatible with each other because people do not give a careful attention to the fashion compatibility of the items. According to lorenenorth.com, the top complaints from customers to fashion stylists include "so many choices, it is hard to mixing and matching clothes" and "I just want a few number of matching outfits instead of buying 30 different items in my closet". 

To solve this problem, I propose AI-based fashion recommendation system for complete outfit based on decision behaviour fashion stylists. From business perspective, the ecommerce fashion industry is a huge industry where the worldwide revenue is measured by 50-60 billion dollars every year. Current ecommerce fashion gigantics such amazon.com do not provide complete outfit recommendation but they do recommend a item of customer's interest based on collaborative filtering, which uses historical data of people who could have similar taste in fashion. Although the collaborative could provide a pair of items to the customers, it can not provide compatible items based on the decision behavior of fashion experts. For these reasons, I envision that my recommendation system can create a win-win opportunity for both fashion websites and their customers, therefore, it can be a game changer that differentiates the fashion websites from its competitors. 

## Setting up the environments

docker build -t api_service -f Dockerfile.api .

docker run -p 8501:8501 -it --runtime=nvidia -v /home/ubuntu/:/root/app api_service