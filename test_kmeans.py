import unittest
import kmeans

class KMeansTests(unittest.TestCase):

    def setUp(self):
        global samples, centroids, k
        samples = [(1,2), (2,3), (3,4)]
        centroids = []
        k = 2

        pass

    def tearDown(self):
        pass

    def test_get_centroids_returns_correct_number(self):
        result = kmeans.get_centroids(k, samples, centroids)

        self.assertEqual(len(result), k)

    def test_get_centroids_returns_all_samples(self):
        k = 3

        result = kmeans.get_centroids(k, samples, centroids)

        self.assertTrue(all([sample in samples for sample in result]))

    def test_kmeans_returns_right_answer(self):
        samples = [(1,1), (1,2), (2,1), (8,1), (8,2),(7,1), (1,8),(2,8),(1,7)]
        k = 3
        centroids = []

        final_centroids = [(1.33, 7.67), (7.67, 1.33), (1.33, 1.33)]

        result = kmeans.kmeans(k, samples, centroids)

        print(result)

        self.assertTrue(all([c in result for c in final_centroids]))