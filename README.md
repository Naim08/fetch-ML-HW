# Fetch Rewards Take-home Exercise - Machine Learning Engineer

## Introduction
Fetch Rewards is a mobile app that allows users to earn points by scanning receipts. The goal of this exercise is to build an end-to-end system that can predict the value of a receipt given an image of a receipt. Their takehome exercise for ML Engineer is to build a model that can forecast monthly scanned receipts on a monthly basis.

Time Spent: 4 hours, actually no. I spent way more time than that.

Solution:

So I took a couple of different approaches, just trying to get a gist of which may lead to more optimal solution. Sin


Extra files in this repo:
seasonal_arima.py: My attempt to find seasonal patterns in the data. I was able to find a seasonal pattern in the data, but it wasnt very strong. I was able to get a seasonal ARIMA model to work, but it wasnt very accurate. I think the data is too noisy to find a strong seasonal pattern.
