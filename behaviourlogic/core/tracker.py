from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        
        # New: History for Chasing Detection
        # {objectID: [(x,y), (x,y), ...]} - Store last N positions
        self.path_history = {} 
        self.history_length = 10 

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.path_history[self.nextObjectID] = [centroid] # Init history
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.path_history:
            del self.path_history[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                
                # Update History
                self.path_history[objectID].append(inputCentroids[col])
                if len(self.path_history[objectID]) > self.history_length:
                    self.path_history[objectID].pop(0)

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

    def detect_chasing(self, threshold_dist=100):
        """
        Detects if one object is following another closely.
        Returns list of tuples (chaser_ID, victim_ID)
        """
        chasing_events = []
        ids = list(self.objects.keys())
        
        if len(ids) < 2: return chasing_events

        for i in range(len(ids)):
            id1 = ids[i]
            if id1 not in self.path_history or len(self.path_history[id1]) < 5: continue
            
            # Vector 1: Movement of person 1 (last point - first point in recent history)
            v1_start = np.array(self.path_history[id1][0])
            v1_end = np.array(self.path_history[id1][-1])
            vec1 = v1_end - v1_start
            speed1 = np.linalg.norm(vec1)

            # Minimum speed check (standing people don't chase)
            if speed1 < 20: continue 

            for j in range(i + 1, len(ids)):
                id2 = ids[j]
                if id2 not in self.path_history or len(self.path_history[id2]) < 5: continue

                v2_start = np.array(self.path_history[id2][0])
                v2_end = np.array(self.path_history[id2][-1])
                vec2 = v2_end - v2_start
                speed2 = np.linalg.norm(vec2)
                
                if speed2 < 20: continue

                # Check Proximity
                curr_dist = dist.euclidean(v1_end, v2_end)
                if curr_dist > threshold_dist: continue

                # Check Vector Alignment (Dot Product)
                # Normalize vectors
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0: continue
                
                cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                
                # If cosine similarity is high (> 0.8), they are moving in same direction
                if cosine_sim > 0.8:
                    # Potential Chasing
                    chasing_events.append((id1, id2))
        
        return chasing_events
