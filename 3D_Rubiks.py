import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import mediapipe as mp
import numpy as np
from rubik_solver import utils
from threading import Thread


class backround:
    def __init__(self, display):
        self.display = display
        self.textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.font = pygame.font.SysFont('arial', 64)

    def draw(self, frame):
        glEnable(GL_TEXTURE_2D)
        # glLoadIdentity()
        frame = cv2.resize(frame, self.display, interpolation=cv2.INTER_AREA)
        # Copy the frame from the webcam into the sender texture
        glBindTexture(GL_TEXTURE_2D, self.textureID)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.display[0], self.display[1], 0, GL_RGB, GL_UNSIGNED_BYTE, frame)

        # Draw texture to screen
        glBegin(GL_QUADS)

        glTexCoord(0, 0)
        glVertex2f(0, 0)

        glTexCoord(1, 0)
        glVertex2f(self.display[0] / 2, 0)

        glTexCoord(1, 1)
        glVertex2f(self.display[0] / 2, self.display[1] / 2)

        glTexCoord(0, 1)
        glVertex2f(0, self.display[1] / 2)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def drawText(self, x, y, text):
        textSurface = self.font.render(text, True, (255, 255, 66, 0)).convert_alpha()
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        glWindowPos2d(x, y)
        glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


class rubiks:
    def __init__(self):
        self.rubiksColor_complete = np.zeros(shape=(6, 3, 3, 3))
        self.rubiksColor_complete[0] += np.array((1, 0, 0))  # RED
        self.rubiksColor_complete[1] += np.array((1, 0.5, 0))  # ORANGE
        self.rubiksColor_complete[2] += np.array((0, 1, 0))  # GREEN
        self.rubiksColor_complete[3] += np.array((0, 0, 1))  # BLUE
        self.rubiksColor_complete[4] += np.array((1, 1, 1))  # WHITE
        self.rubiksColor_complete[5] += np.array((1, 1, 0))  # YELLOW

        self.rubiksColor = np.copy(self.rubiksColor_complete)

    def ColorFaceSwap(self, face, direction):
        color_buffer = np.zeros(shape=(3, 3))

        if face == 0 and direction == -1:
            self.rubiksColor[0] = self.rubiksColor[0].swapaxes(0, 1)[:, [2, 1, 0], :]
            color_buffer += self.rubiksColor[2, :, 0]
            self.rubiksColor[2, :, 0] = np.flip(self.rubiksColor[4, 0, :], 0)  # W->B
            self.rubiksColor[4, 0, :] = self.rubiksColor[3, :, 0]  # G->W
            self.rubiksColor[3, :, 0] = np.flip(self.rubiksColor[5, 0, :], 0)  # Y->G
            self.rubiksColor[5, 0, :] = color_buffer  # B->Y

        elif face == 0 and direction == 1:
            self.rubiksColor[0] = self.rubiksColor[0].swapaxes(0, 1)[[2, 1, 0], :, :]
            color_buffer += self.rubiksColor[5, 0, :]
            self.rubiksColor[5, 0, :] = np.flip(self.rubiksColor[3, :, 0], 0)
            self.rubiksColor[3, :, 0] = self.rubiksColor[4, 0, :]
            self.rubiksColor[4, 0, :] = np.flip(self.rubiksColor[2, :, 0], 0)
            self.rubiksColor[2, :, 0] = color_buffer

        elif face == 1 and direction == -1:
            self.rubiksColor[1] = self.rubiksColor[1].swapaxes(0, 1)[[2, 1, 0], :, :]
            color_buffer += self.rubiksColor[5, 2, :]
            self.rubiksColor[5, 2, :] = np.flip(self.rubiksColor[3, :, 2], 0)
            self.rubiksColor[3, :, 2] = self.rubiksColor[4, 2, :]
            self.rubiksColor[4, 2, :] = np.flip(self.rubiksColor[2, :, 2], 0)
            self.rubiksColor[2, :, 2] = color_buffer

        elif face == 1 and direction == 1:
            self.rubiksColor[1] = self.rubiksColor[1].swapaxes(0, 1)[:, [2, 1, 0], :]
            color_buffer += self.rubiksColor[2, :, 2]
            self.rubiksColor[2, :, 2] = np.flip(self.rubiksColor[4, 2, :], 0)  # W->B
            self.rubiksColor[4, 2, :] = self.rubiksColor[3, :, 2]  # G->W
            self.rubiksColor[3, :, 2] = np.flip(self.rubiksColor[5, 2, :], 0)  # Y->G
            self.rubiksColor[5, 2, :] = color_buffer  # B->Y

        elif face == 2 and direction == -1:
            self.rubiksColor[2] = self.rubiksColor[2].swapaxes(0, 1)[:, [2, 1, 0], :]
            color_buffer += self.rubiksColor[4, :, 0]
            self.rubiksColor[4, :, 0] = np.flip(self.rubiksColor[0, 0, :], 0)
            self.rubiksColor[0, 0, :] = self.rubiksColor[5, :, 0]
            self.rubiksColor[5, :, 0] = np.flip(self.rubiksColor[1, 0, :], 0)
            self.rubiksColor[1, 0, :] = color_buffer

        elif face == 2 and direction == 1:
            self.rubiksColor[2] = self.rubiksColor[2].swapaxes(0, 1)[[2, 1, 0], :, :]
            color_buffer += self.rubiksColor[1, 0, :]
            self.rubiksColor[1, 0, :] = np.flip(self.rubiksColor[5, :, 0], 0)
            self.rubiksColor[5, :, 0] = self.rubiksColor[0, 0, :]
            self.rubiksColor[0, 0, :] = np.flip(self.rubiksColor[4, :, 0], 0)
            self.rubiksColor[4, :, 0] = color_buffer

        elif face == 3 and direction == -1:
            self.rubiksColor[3] = self.rubiksColor[3].swapaxes(0, 1)[[2, 1, 0], :, :]
            color_buffer += self.rubiksColor[1, 2, :]
            self.rubiksColor[1, 2, :] = np.flip(self.rubiksColor[5, :, 2], 0)
            self.rubiksColor[5, :, 2] = self.rubiksColor[0, 2, :]
            self.rubiksColor[0, 2, :] = np.flip(self.rubiksColor[4, :, 2], 0)
            self.rubiksColor[4, :, 2] = color_buffer

        elif face == 3 and direction == 1:
            self.rubiksColor[3] = self.rubiksColor[3].swapaxes(0, 1)[:, [2, 1, 0], :]
            color_buffer += self.rubiksColor[4, :, 2]
            self.rubiksColor[4, :, 2] = np.flip(self.rubiksColor[0, 2, :], 0)
            self.rubiksColor[0, 2, :] = self.rubiksColor[5, :, 2]
            self.rubiksColor[5, :, 2] = np.flip(self.rubiksColor[1, 2, :], 0)
            self.rubiksColor[1, 2, :] = color_buffer

        elif face == 4 and direction == -1:
            self.rubiksColor[4] = self.rubiksColor[4].swapaxes(0, 1)[:, [2, 1, 0], :]
            color_buffer += self.rubiksColor[0, :, 0]
            self.rubiksColor[0, :, 0] = np.flip(self.rubiksColor[2, 0, :], 0)
            self.rubiksColor[2, 0, :] = self.rubiksColor[1, :, 0]
            self.rubiksColor[1, :, 0] = np.flip(self.rubiksColor[3, 0, :], 0)
            self.rubiksColor[3, 0, :] = color_buffer

        elif face == 4 and direction == 1:
            self.rubiksColor[4] = self.rubiksColor[4].swapaxes(0, 1)[[2, 1, 0], :, :]
            color_buffer += self.rubiksColor[3, 0, :]
            self.rubiksColor[3, 0, :] = np.flip(self.rubiksColor[1, :, 0], 0)
            self.rubiksColor[1, :, 0] = self.rubiksColor[2, 0, :]
            self.rubiksColor[2, 0, :] = np.flip(self.rubiksColor[0, :, 0], 0)
            self.rubiksColor[0, :, 0] = color_buffer

        elif face == 5 and direction == -1:
            self.rubiksColor[5] = self.rubiksColor[5].swapaxes(0, 1)[[2, 1, 0], :, :]
            color_buffer += self.rubiksColor[3, 2, :]
            self.rubiksColor[3, 2, :] = np.flip(self.rubiksColor[1, :, 2], 0)
            self.rubiksColor[1, :, 2] = self.rubiksColor[2, 2, :]
            self.rubiksColor[2, 2, :] = np.flip(self.rubiksColor[0, :, 2], 0)
            self.rubiksColor[0, :, 2] = color_buffer

        elif face == 5 and direction == 1:
            self.rubiksColor[5] = self.rubiksColor[5].swapaxes(0, 1)[:, [2, 1, 0], :]
            color_buffer += self.rubiksColor[0, :, 2]
            self.rubiksColor[0, :, 2] = np.flip(self.rubiksColor[2, 2, :], 0)
            self.rubiksColor[2, 2, :] = self.rubiksColor[1, :, 2]
            self.rubiksColor[1, :, 2] = np.flip(self.rubiksColor[3, 2, :], 0)
            self.rubiksColor[3, 2, :] = color_buffer

    def Reset(self):
        self.rubiksColor = np.copy(self.rubiksColor_complete)

    def Mix_rubiks(self, k):
        return np.stack((np.random.randint(0, 6, k), np.random.choice([-1, 1], k)), axis=1)

    def Solveur(self):
        def color_code(color):
            if np.array_equiv(color, np.array([1, 0, 0])):
                return "r"
            elif np.array_equiv(color, np.array([1, 0.5, 0])):
                return "o"
            elif np.array_equiv(color, np.array([0, 0, 1])):
                return "b"
            elif np.array_equiv(color, np.array([0, 1, 0])):
                return "g"
            elif np.array_equiv(color, np.array([1, 1, 0])):
                return "y"
            else:
                return "w"

        def matrix_rotate(mat, n):
            for i in range(n):
                mat = mat.swapaxes(0, 1)[:, [2, 1, 0], :]
            return mat

        color_matrix = np.copy(self.rubiksColor)

        color_matrix[5] = matrix_rotate(color_matrix[5], 2)
        color_matrix[0] = matrix_rotate(color_matrix[0], 2).swapaxes(0, 1)
        color_matrix[4] = matrix_rotate(color_matrix[4], 3).swapaxes(0, 1)
        color_matrix[3] = matrix_rotate(color_matrix[3], 2)
        color_matrix[2] = matrix_rotate(color_matrix[2], 1).swapaxes(0, 1)
        color_matrix[1] = matrix_rotate(color_matrix[1], 3)

        color_string = ""
        for face in [5, 3, 0, 2, 1, 4]:
            for i in range(3):
                for j in range(3):
                    color_string += color_code(color_matrix[face, i, j])
        solving_steps = utils.solve(color_string, 'Kociemba')
        moove_face = {"F": 0, "B": 1, "L": 3, "R": 2, "U": 5, "D": 4}

        mooves = list()

        for step in solving_steps:

            if len(step.raw) > 1:
                if step.raw[1] == "2":
                    mooves.append([moove_face[step.raw[0]], 1])
                    mooves.append([moove_face[step.raw[0]], 1])
                else:
                    mooves.append([moove_face[step.raw[0]], -1])
            else:
                mooves.append([moove_face[step.raw[0]], 1])

        return mooves

    def Complete(self):
        return np.array_equal(self.rubiksColor, self.rubiksColor_complete)


class rubiks_display:

    def __init__(self, Rubiks):
        self.Rubiks = Rubiks
        self.vertices = np.array(
            ((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)))
        self.edges = np.array(
            ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)))
        self.faces = np.array(((0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3), (1, 5, 6, 2), (1, 5, 4, 0), (2, 6, 7, 3)))

        self.rotation_face = -1
        self.angle = 0
        self.moove_buffer = list()
        self.rotate = "NO"
        self.x0, self.y0, self.cX, self.cY = 0, 0, 0, 0
        self.upFace = 1
        self.modelMatrix = 0

    def Cube(self, offset, faces_color, vertice_color=(0, 0, 0), linewidth=4):
        glBegin(GL_QUADS)
        for face, color in zip(self.faces, faces_color):
            glColor3fv(color)
            for vertex in face:
                glVertex3fv(self.vertices[vertex] + offset)
        glEnd()

        glLineWidth(linewidth)
        glBegin(GL_LINES)
        for edge in self.edges:
            for vertex in edge:
                glColor3fv(vertice_color)
                glVertex3fv(self.vertices[vertex] + offset)
        glEnd()

    def DrawLowerCube(self, face):
        lowerZone = np.ones(shape=(3, 3, 3))
        if face == 0:
            lowerZone[0, :, :] *= 0
        if face == 1:
            lowerZone[2, :, :] *= 0
        if face == 2:
            lowerZone[:, 0, :] *= 0
        if face == 3:
            lowerZone[:, 2, :] *= 0
        if face == 4:
            lowerZone[:, :, 0] *= 0
        if face == 5:
            lowerZone[:, :, 2] *= 0

        for i in range(3):  # face 1,0
            for j in range(3):  # face 2,3
                for k in range(3):  # face 4,5
                    if lowerZone[i, j, k]:
                        faces_color = np.zeros(shape=(6, 3))
                        if i == 2:
                            faces_color[1] = self.Rubiks.rubiksColor[1, j, k]
                        if i == 0:
                            faces_color[0] = self.Rubiks.rubiksColor[0, j, k]
                        if j == 2:
                            faces_color[3] = self.Rubiks.rubiksColor[3, k, i]
                        if j == 0:
                            faces_color[2] = self.Rubiks.rubiksColor[2, k, i]
                        if k == 2:
                            faces_color[5] = self.Rubiks.rubiksColor[5, i, j]
                        if k == 0:
                            faces_color[4] = self.Rubiks.rubiksColor[4, i, j]
                        offset = np.array([2 * (i - 1), 2 * (j - 1), 2 * (k - 1)])
                        self.Cube(offset, faces_color)

    def DrawUpperCube(self, face):
        lowerZone = np.ones(shape=(3, 3, 3))
        if face == 0:
            lowerZone[0, :, :] *= 0
        if face == 1:
            lowerZone[2, :, :] *= 0
        if face == 2:
            lowerZone[:, 0, :] *= 0
        if face == 3:
            lowerZone[:, 2, :] *= 0
        if face == 4:
            lowerZone[:, :, 0] *= 0
        if face == 5:
            lowerZone[:, :, 2] *= 0

        for i in range(3):  # face 1,0
            for j in range(3):  # face 2,3
                for k in range(3):  # face 4,5
                    if lowerZone[i, j, k] == 0:
                        faces_color = np.zeros(shape=(6, 3))
                        if i == 2:
                            faces_color[1] = self.Rubiks.rubiksColor[1, j, k]
                        if i == 0:
                            faces_color[0] = self.Rubiks.rubiksColor[0, j, k]
                        if j == 2:
                            faces_color[3] = self.Rubiks.rubiksColor[3, k, i]
                        if j == 0:
                            faces_color[2] = self.Rubiks.rubiksColor[2, k, i]
                        if k == 2:
                            faces_color[5] = self.Rubiks.rubiksColor[5, i, j]
                        if k == 0:
                            faces_color[4] = self.Rubiks.rubiksColor[4, i, j]
                        offset = np.array([2 * (i - 1), 2 * (j - 1), 2 * (k - 1)])
                        self.Cube(offset, faces_color, (0.7, 0.7, 0.7), 7)

    def DrawRubiks(self):
        glEnable(GL_DEPTH_TEST)
        glRotatef(self.cX, 0, 1, 0)  # on applique les rotations a la matrice du gl
        glRotatef(self.cY, 1, 0, 0)

        glMultMatrixf(self.modelMatrix)

        self.modelMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)  # on recharge l'ancienne matrice d'orientation
        glLoadIdentity()  # on reinitialise la matrice de position

        glTranslatef(0, 0, -20)
        glMultMatrixf(self.modelMatrix)

        glEnable(GL_DEPTH_TEST)

        upFace = self.rotation_face
        if not self.in_animation():
            upFace = self.Upper_face()

        self.DrawLowerCube(upFace)  # on dessine la partie inférieur du rubiks

        dtetha = 15

        if self.rotation_face == -1:
            if len(self.moove_buffer) == 0:
                if self.rotate == "LEFT":
                    self.rotation_face = self.Upper_face()
                    self.angle = 5

                elif self.rotate == "RIGHT":
                    self.rotation_face = self.Upper_face()
                    self.angle = -5
            else:
                self.rotation_face = self.moove_buffer[0][0]
                self.angle = 5 * self.moove_buffer[0][1]

        else:
            self.angle += np.sign(self.angle) * np.sin((np.sign(self.angle) + 30) * np.pi / 180) * dtetha
            if np.abs(self.angle) >= 90:
                self.Rubiks.ColorFaceSwap(self.rotation_face, np.sign(self.angle))
                self.angle = 0
                self.rotation_face_animation(self.rotation_face, self.angle)
                self.rotation_face = -1
                if len(self.moove_buffer) > 0: self.moove_buffer = self.moove_buffer[1:]

                if self.Rubiks.Complete():
                    print("Congratulation !")
            else:
                self.rotation_face_animation(self.rotation_face, self.angle)
        modelMatrix2 = glGetFloatv(GL_MODELVIEW_MATRIX)
        glLoadIdentity()  # on reinitialise la matrice de position

        glMultMatrixf(modelMatrix2)
        self.DrawUpperCube(upFace)  # on dessine la partie supérieur du rubiks

    def rotation_face_animation(self, face, angle):
        if face == 0:
            glRotatef(angle, 1, 0, 0)
        elif face == 1:
            glRotatef(-angle, 1, 0, 0)
        elif face == 2:
            glRotatef(angle, 0, 1, 0)
        elif face == 3:
            glRotatef(-angle, 0, 1, 0)
        elif face == 4:
            glRotatef(angle, 0, 0, 1)
        elif face == 5:
            glRotatef(-angle, 0, 0, 1)

    def Controls(self, x, y, rotate):
        self.rotate = rotate
        self.cX += (x - self.x0) * 70 - self.cX / 10
        self.cY += (y - self.y0) * 70 - self.cY / 10
        self.x0 = x
        self.y0 = y

        if np.abs(self.cX) < 1:
            self.cX = 0
        if np.abs(self.cY) < 1:
            self.cY = 0

    def Upper_face(self):
        # roll = np.arctan2(modelMatrix[2, 1], modelMatrix[2, 2]) * 180 / np.pi
        # yaw = np.arctan2(-modelMatrix[2, 0], (modelMatrix[2, 1] ** 2 + modelMatrix[2, 2] ** 2) ** 0.5) * 180 / np.pi
        # pitch = np.arctan2(-modelMatrix[1, 0], modelMatrix[0, 0]) * 180 / np.pi
        # print("pitch :", format(pitch,".1f"), "\t roll :", format(roll,".1f"), "\t yaw :", format(yaw,".1f"))
        Vector = np.array([0, 1, 0.2])
        alignement_factor = np.dot(self.modelMatrix[:3, :3], Vector)

        max_direction = np.argmax(np.abs(alignement_factor))

        index_upper = int((max_direction + 0.5 * (1 + np.sign(alignement_factor[max_direction]))) * np.sign(
            alignement_factor[max_direction]) + 2)

        upper_faces_name = ["White", "Blue", "Red", "Orange", "Green", "Yellow"]

        upper_faces_id = np.array([4, 2, 0, 1, 3, 5])
        # print(upper_faces_id[index_upper])
        return upper_faces_id[index_upper]

    def in_animation(self):
        return len(self.moove_buffer) > 0 or self.rotate != "NO"


class hand_gesture:

    def __init__(self):
        self.gesture = (0, 0, "NO", None)
        self.end_thread = False

    def Detection(self):

        def inside_triangle(A, B, C, M):
            Area = 0.5 * (-B[1] * C[0] + A[1] * (-B[0] + C[0]) + A[0] * (B[1] - C[1]) + B[0] * C[1])
            s = 1 / (2 * Area) * (A[1] * C[0] - A[0] * C[1] + (C[1] - A[1]) * M[0] + (A[0] - C[0]) * M[1])
            t = 1 / (2 * Area) * (A[0] * B[1] - A[1] * B[0] + (A[1] - B[1]) * M[0] + (B[0] - A[0]) * M[1])
            if s > 0 and t > 0 and 1 - s - t > 0:
                return True
            else:
                return False

        cap = cv2.VideoCapture(0)

        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils
        Thumb_was_outside = True
        Pinky_was_outside = True

        while not self.end_thread:
            success, img = cap.read()
            if success:
                img = cv2.flip(img, 1)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)
                rotation = "NO"
                xR, yR = 0, 0

                if results.multi_hand_landmarks:
                    handlms = results.multi_hand_landmarks
                    n_hands = len(handlms)
                    R = True

                    for i in range(n_hands):
                        if results.multi_handedness[i].classification[0].label == "Right" and R:
                            R = False
                            xR = handlms[i].landmark[9].x
                            yR = handlms[i].landmark[9].y
                            A = [handlms[i].landmark[0].x, handlms[i].landmark[0].y]
                            B = [handlms[i].landmark[5].x, handlms[i].landmark[5].y]
                            C = [handlms[i].landmark[18].x, handlms[i].landmark[18].y]
                            Thumb = [handlms[i].landmark[4].x, handlms[i].landmark[4].y]
                            Pinky = [handlms[i].landmark[20].x, handlms[i].landmark[20].y]
                            if inside_triangle(A, B, C, Thumb):
                                if Thumb_was_outside:
                                    Thumb_was_outside = False
                                    rotation = "RIGHT"


                            elif inside_triangle(A, B, C, Pinky):
                                if Pinky_was_outside:
                                    Pinky_was_outside = False
                                    rotation = "LEFT"
                            else:
                                Thumb_was_outside = True
                                Pinky_was_outside = True

                            mpDraw.draw_landmarks(img, handlms[i], mpHands.HAND_CONNECTIONS)
            else:
                img = None
            self.gesture = (xR, yR, rotation, img)
            pygame.time.wait(5)


def main():
    pygame.init()

    Rubiks = rubiks()
    Rubiks_display = rubiks_display(Rubiks)

    Hand = hand_gesture()

    Detector_thread = Thread(target=Hand.Detection)
    Detector_thread.start()

    clock = pygame.time.Clock()

    display = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL | RESIZABLE | SCALED)
    screen = display.get_size()
    pygame.display.set_caption("7dric Rubik's Cube")

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (screen[0] / screen[1]), 0.1, 500)
    glMatrixMode(GL_MODELVIEW)
    Rubiks_display.modelMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glOrtho(0, screen[0], screen[1], 0, 1, -1)
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.1, 0.1, 0.2, 1)

    Backround = backround(screen)

    while True:

        glPushMatrix()
        glLoadIdentity()

        # TODO define rotation face quand buffer vide

        # getting the thread value:
        x, y, rotate, img = Hand.gesture

        Rubiks_display.Controls(x, y, rotate)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                Hand.end_thread = True
                Detector_thread.join(timeout=10)
                pygame.quit()
                sys.exit()

            elif event.type == pygame.VIDEORESIZE:
                screen = [event.w, event.h]

            elif event.type == pygame.KEYDOWN and not Rubiks_display.in_animation():
                if event.key == pygame.K_r:
                    Rubiks.reset()

                elif event.key == pygame.K_RIGHT:
                    Rubiks_display.rotate = "RIGHT"

                elif event.key == pygame.K_LEFT:
                    Rubiks_display.rotate = "LEFT"

                elif event.key == pygame.K_m:
                    Rubiks_display.moove_buffer = Rubiks.Mix_rubiks(40)

                elif event.key == pygame.K_s:
                    Rubiks_display.moove_buffer = Rubiks.Solveur()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        Rubiks_display.DrawRubiks()

        glPopMatrix()

        text = str(int(clock.get_fps())) + "FPS"
        Backround.drawText(20, 20, text)

        pygame.display.flip()
        # if img is not None : cv2.imshow("Webcam",img)
        clock.tick()
        pygame.time.wait(5)


if __name__ == "__main__":
    main()

# Grand merci mev pour l'idée
