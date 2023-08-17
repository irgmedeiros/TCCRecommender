# Study about Collaborative Recommender System - 2013

Python implementation of collaborative filtering recommendation algorithms with a focus on evaluating different similarity measures. This work was done as part of a computer science monograph at UFPE by Igor Rafael Guimarães Medeiros under the supervision of Prof. Ricardo Bastos Cavalcante Prudêncio. The paper is called “Estudo sobre Sistemas de Recomendação Colaborativos” and it's a study review of main recommendations systems approaches with a focus on evaluating different similarity measures on collaborative filtering.

## Purpose

The goal of this project is to implement and compare various similarity algorithms for collaborative filtering recommenders. A key contribution is the Python implementation of adjusted cosine similarity (the first one available on GitHub at the time of the study) and its performance compared to traditional similarity algorithms. 

## Background

Collaborative filtering recommenders are widely used, but the choice of similarity metric can impact performance. This work conducts experiments on real-world datasets to evaluate different similarity approaches.

## Datasets

Movies, books, and music ratings datasets are included to allow reproducibility. Specific datasets used in the original experiments include MovieLens ratings.

## Key Findings

Experimental results showed that adjusted cosine similarity outperformed traditional cosine and correlation for calculating user similarities. Adjusted cosine was chosen as the default similarity algorithm for the recommenders due to this improved performance.

## Publication

The key results and approaches were published by Igor Medeiros in the monograph _“Estudos sobre Sistemas de Recomendação Colaborativos”_, originally at [cin.ufpe.br](https://www.cin.ufpe.br/~tg/2012-2/irgm.pdf) but also available [here](https://github.com/irgmedeiros/TCCRecommender/blob/master/Estudo%20sobre%20Sistemas%20de%20Recomenda%C3%A7%C3%A3o%20Colaborativos%20by%20Igor%20Medeiros.pdf).

### Author

Igor Rafael Guimarães Medeiros, Bachelor in Computer Science at Federal University of Pernambuco (UFPE).

Contact: [irgmedeiros@proton.me](mailto:irgmedeiros@proton.me)

### Supervisor

PhD. Ricardo Bastos Cavalcante Prudêncio, Professor of computer science at UFPE.

Contact: [rbcp@cin.ufpe.br](mailto:rbcp@cin.ufpe.br)

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

By including a Citation section with a BibTeX entry, it properly attributes the original work if someone wants to reference it academically. And stating the license upfront following Citation makes it clear how others can use and modify the code legally.
