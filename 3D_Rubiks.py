import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import cv2
import mediapipe as mp
import numpy as np
from rubik_solver import utils
from threading import Thread


def Cube(vertices, edges, faces, faces_color, vertice_color=(0, 0, 0), linewidth=4):
    glBegin(GL_QUADS)
    for face, color in zip(faces, faces_color):
        glColor3fv(color)
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    glLineWidth(linewidth)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv(vertice_color)
            glVertex3fv(vertices[vertex])
    glEnd()


def RubixCube():
    for i in range(3):  # face 1,0
        for j in range(3):  # face 2,3
            for k in range(3):  # face 4,5
                faces_color = np.zeros(shape=(6, 3))
                if i == 2:
                    faces_color[1] = rubiksColor[1, j, k]
                if i == 0:
                    faces_color[0] = rubiksColor[0, j, k]
                if j == 2:
                    faces_color[3] = rubiksColor[3, k, i]
                if j == 0:
                    faces_color[2] = rubiksColor[2, k, i]
                if k == 2:
                    faces_color[5] = rubiksColor[5, i, j]
                if k == 0:
                    faces_color[4] = rubiksColor[4, i, j]
                offset = np.array([2 * (i - 1), 2 * (j - 1), 2 * (k - 1)])
                Cube(vertices + offset, edges, faces, faces_color)


def DrawLowerCube(face):
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
                        faces_color[1] = rubiksColor[1, j, k]
                    if i == 0:
                        faces_color[0] = rubiksColor[0, j, k]
                    if j == 2:
                        faces_color[3] = rubiksColor[3, k, i]
                    if j == 0:
                        faces_color[2] = rubiksColor[2, k, i]
                    if k == 2:
                        faces_color[5] = rubiksColor[5, i, j]
                    if k == 0:
                        faces_color[4] = rubiksColor[4, i, j]
                    offset = np.array([2 * (i - 1), 2 * (j - 1), 2 * (k - 1)])
                    Cube(vertices + offset, edges, faces, faces_color)


def DrawUpperCube(face):
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
                        faces_color[1] = rubiksColor[1, j, k]
                    if i == 0:
                        faces_color[0] = rubiksColor[0, j, k]
                    if j == 2:
                        faces_color[3] = rubiksColor[3, k, i]
                    if j == 0:
                        faces_color[2] = rubiksColor[2, k, i]
                    if k == 2:
                        faces_color[5] = rubiksColor[5, i, j]
                    if k == 0:
                        faces_color[4] = rubiksColor[4, i, j]
                    offset = np.array([2 * (i - 1), 2 * (j - 1), 2 * (k - 1)])
                    Cube(vertices + offset, edges, faces, faces_color, (0.7, 0.7, 0.7), 7)


def ColorFaceSwap(face, direction):
    color_buffer = np.zeros(shape=(3, 3))

    if face == 0 and direction == -1:
        rubiksColor[0] = rubiksColor[0].swapaxes(0, 1)[:, [2, 1, 0], :]
        color_buffer += rubiksColor[2, :, 0]
        rubiksColor[2, :, 0] = np.flip(rubiksColor[4, 0, :], 0)  # W->B
        rubiksColor[4, 0, :] = rubiksColor[3, :, 0]  # G->W
        rubiksColor[3, :, 0] = np.flip(rubiksColor[5, 0, :], 0)  # Y->G
        rubiksColor[5, 0, :] = color_buffer  # B->Y

    elif face == 0 and direction == 1:
        rubiksColor[0] = rubiksColor[0].swapaxes(0, 1)[[2, 1, 0], :, :]
        color_buffer += rubiksColor[5, 0, :]
        rubiksColor[5, 0, :] = np.flip(rubiksColor[3, :, 0], 0)
        rubiksColor[3, :, 0] = rubiksColor[4, 0, :]
        rubiksColor[4, 0, :] = np.flip(rubiksColor[2, :, 0], 0)
        rubiksColor[2, :, 0] = color_buffer

    elif face == 1 and direction == -1:
        rubiksColor[1] = rubiksColor[1].swapaxes(0, 1)[[2, 1, 0], :, :]
        color_buffer += rubiksColor[5, 2, :]
        rubiksColor[5, 2, :] = np.flip(rubiksColor[3, :, 2], 0)
        rubiksColor[3, :, 2] = rubiksColor[4, 2, :]
        rubiksColor[4, 2, :] = np.flip(rubiksColor[2, :, 2], 0)
        rubiksColor[2, :, 2] = color_buffer

    elif face == 1 and direction == 1:
        rubiksColor[1] = rubiksColor[1].swapaxes(0, 1)[:, [2, 1, 0], :]
        color_buffer += rubiksColor[2, :, 2]
        rubiksColor[2, :, 2] = np.flip(rubiksColor[4, 2, :], 0)  # W->B
        rubiksColor[4, 2, :] = rubiksColor[3, :, 2]  # G->W
        rubiksColor[3, :, 2] = np.flip(rubiksColor[5, 2, :], 0)  # Y->G
        rubiksColor[5, 2, :] = color_buffer  # B->Y

    elif face == 2 and direction == -1:
        rubiksColor[2] = rubiksColor[2].swapaxes(0, 1)[:, [2, 1, 0], :]
        color_buffer += rubiksColor[4, :, 0]
        rubiksColor[4, :, 0] = np.flip(rubiksColor[0, 0, :], 0)
        rubiksColor[0, 0, :] = rubiksColor[5, :, 0]
        rubiksColor[5, :, 0] = np.flip(rubiksColor[1, 0, :], 0)
        rubiksColor[1, 0, :] = color_buffer

    elif face == 2 and direction == 1:
        rubiksColor[2] = rubiksColor[2].swapaxes(0, 1)[[2, 1, 0], :, :]
        color_buffer += rubiksColor[1, 0, :]
        rubiksColor[1, 0, :] = np.flip(rubiksColor[5, :, 0], 0)
        rubiksColor[5, :, 0] = rubiksColor[0, 0, :]
        rubiksColor[0, 0, :] = np.flip(rubiksColor[4, :, 0], 0)
        rubiksColor[4, :, 0] = color_buffer

    elif face == 3 and direction == -1:
        rubiksColor[3] = rubiksColor[3].swapaxes(0, 1)[[2, 1, 0], :, :]
        color_buffer += rubiksColor[1, 2, :]
        rubiksColor[1, 2, :] = np.flip(rubiksColor[5, :, 2], 0)
        rubiksColor[5, :, 2] = rubiksColor[0, 2, :]
        rubiksColor[0, 2, :] = np.flip(rubiksColor[4, :, 2], 0)
        rubiksColor[4, :, 2] = color_buffer

    elif face == 3 and direction == 1:
        rubiksColor[3] = rubiksColor[3].swapaxes(0, 1)[:, [2, 1, 0], :]
        color_buffer += rubiksColor[4, :, 2]
        rubiksColor[4, :, 2] = np.flip(rubiksColor[0, 2, :], 0)
        rubiksColor[0, 2, :] = rubiksColor[5, :, 2]
        rubiksColor[5, :, 2] = np.flip(rubiksColor[1, 2, :], 0)
        rubiksColor[1, 2, :] = color_buffer

    elif face == 4 and direction == -1:
        rubiksColor[4] = rubiksColor[4].swapaxes(0, 1)[:, [2, 1, 0], :]
        color_buffer += rubiksColor[0, :, 0]
        rubiksColor[0, :, 0] = np.flip(rubiksColor[2, 0, :], 0)
        rubiksColor[2, 0, :] = rubiksColor[1, :, 0]
        rubiksColor[1, :, 0] = np.flip(rubiksColor[3, 0, :], 0)
        rubiksColor[3, 0, :] = color_buffer

    elif face == 4 and direction == 1:
        rubiksColor[4] = rubiksColor[4].swapaxes(0, 1)[[2, 1, 0], :, :]
        color_buffer += rubiksColor[3, 0, :]
        rubiksColor[3, 0, :] = np.flip(rubiksColor[1, :, 0], 0)
        rubiksColor[1, :, 0] = rubiksColor[2, 0, :]
        rubiksColor[2, 0, :] = np.flip(rubiksColor[0, :, 0], 0)
        rubiksColor[0, :, 0] = color_buffer

    elif face == 5 and direction == -1:
        rubiksColor[5] = rubiksColor[5].swapaxes(0, 1)[[2, 1, 0], :, :]
        color_buffer += rubiksColor[3, 2, :]
        rubiksColor[3, 2, :] = np.flip(rubiksColor[1, :, 2], 0)
        rubiksColor[1, :, 2] = rubiksColor[2, 2, :]
        rubiksColor[2, 2, :] = np.flip(rubiksColor[0, :, 2], 0)
        rubiksColor[0, :, 2] = color_buffer

    elif face == 5 and direction == 1:
        rubiksColor[5] = rubiksColor[5].swapaxes(0, 1)[:, [2, 1, 0], :]
        color_buffer += rubiksColor[0, :, 2]
        rubiksColor[0, :, 2] = np.flip(rubiksColor[2, 2, :], 0)
        rubiksColor[2, 2, :] = rubiksColor[1, :, 2]
        rubiksColor[1, :, 2] = np.flip(rubiksColor[3, 2, :], 0)
        rubiksColor[3, 2, :] = color_buffer


def Upper_face(matrix):
    # roll = np.arctan2(modelMatrix[2, 1], modelMatrix[2, 2]) * 180 / np.pi
    # yaw = np.arctan2(-modelMatrix[2, 0], (modelMatrix[2, 1] ** 2 + modelMatrix[2, 2] ** 2) ** 0.5) * 180 / np.pi
    # pitch = np.arctan2(-modelMatrix[1, 0], modelMatrix[0, 0]) * 180 / np.pi
    # print("pitch :", format(pitch,".1f"), "\t roll :", format(roll,".1f"), "\t yaw :", format(yaw,".1f"))
    Vector = np.array([0, 1, 0.2])
    alignement_factor = np.dot(matrix[:3, :3], Vector)

    max_direction = np.argmax(np.abs(alignement_factor))

    index_upper = int((max_direction + 0.5 * (1 + np.sign(alignement_factor[max_direction]))) * np.sign(
        alignement_factor[max_direction]) + 2)

    upper_faces_name = ["White", "Blue", "Red", "Orange", "Green", "Yellow"]

    upper_faces_id = np.array([4, 2, 0, 1, 3, 5])
    # print(upper_faces_id[index_upper])
    return upper_faces_id[index_upper]


def inside_triangle(A, B, C, M):
    Area = 0.5 * (-B[1] * C[0] + A[1] * (-B[0] + C[0]) + A[0] * (B[1] - C[1]) + B[0] * C[1])
    s = 1 / (2 * Area) * (A[1] * C[0] - A[0] * C[1] + (C[1] - A[1]) * M[0] + (A[0] - C[0]) * M[1])
    t = 1 / (2 * Area) * (A[0] * B[1] - A[1] * B[0] + (A[1] - B[1]) * M[0] + (B[0] - A[0]) * M[1])
    if s > 0 and t > 0 and 1 - s - t > 0:
        return True
    else:
        return False


def rotation_face_animation(face, angle):
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


def Hand_gesture():
    global gesture

    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    Thumb_was_outside = True
    Pinky_was_outside = True
    while (True):
        success, img = cap.read()
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
                    xR = handlms[i].landmark[8].x
                    yR = handlms[i].landmark[8].y
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
        #cv2.imshow("Image", img)
        gesture = (xR, yR, rotation)


def Mix_rubiks(k):
    return np.stack((np.random.randint(0, 6, k), np.random.choice([-1, 1], k)), axis=1)


def solveur():
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

    color_matrix = np.copy(rubiksColor)

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


def Main():
    rubiksColor_complete = np.zeros(shape=(6, 3, 3, 3))
    rubiksColor_complete[0] += np.array((1, 0, 0))  # RED
    rubiksColor_complete[1] += np.array((1, 0.5, 0))  # ORANGE
    rubiksColor_complete[2] += np.array((0, 1, 0))  # GREEN
    rubiksColor_complete[3] += np.array((0, 0, 1))  # BLUE
    rubiksColor_complete[4] += np.array((1, 1, 1))  # WHITE
    rubiksColor_complete[5] += np.array((1, 1, 0))  # YELLOW

    global vertices, edges, faces, rubiksColor
    vertices = np.array(
        ((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)))
    edges = np.array(((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)))
    faces = np.array(((0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3), (1, 5, 6, 2), (1, 5, 4, 0), (2, 6, 7, 3)))
    rubiksColor = np.copy(rubiksColor_complete)

    global gesture
    Detector_thread = Thread(target=Hand_gesture)
    Detector_thread.start()

    pygame.init()

    clock = pygame.time.Clock()
    screen = (800, 600)

    pygame.display.set_mode(screen, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE | pygame.SCALED)
    pygame.display.set_caption("7dric Rubik's Cube")
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (screen[0] / screen[1]), 0.1, 500)

    glMatrixMode(GL_MODELVIEW)
    modelMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
    glEnable(GL_DEPTH_TEST)

    xR0, yR0, cX, cY, angle = 0, 0, 0, 0, 0
    gesture = (0, 0, "NO")
    rotation_face = -1

    moove_buffer = list()

    while True:

        glPushMatrix()
        glLoadIdentity()

        upFace = rotation_face
        if rotation_face == -1:
            upFace = Upper_face(modelMatrix)

        # getting the thread value:
        xR, yR, rotate = gesture

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and len(moove_buffer) == 0:
                    rubiksColor = np.copy(rubiksColor_complete)

                elif event.key == pygame.K_RIGHT:
                    rotate = "RIGHT"

                elif event.key == pygame.K_LEFT:
                    rotate = "LEFT"

                elif event.key == pygame.K_m and len(moove_buffer) == 0:
                    moove_buffer = Mix_rubiks(40)

                elif event.key == pygame.K_s and len(moove_buffer) == 0:
                    moove_buffer = solveur()

        cX += (xR - xR0) * 70 - cX / 10
        cY += (yR - yR0) * 70 - cY / 10
        xR0 = xR
        yR0 = yR

        if np.abs(cX) < 1:
            cX = 0
        if np.abs(cY) < 1:
            cY = 0

        glClearColor(0.1, 0.1, 0.2, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glRotatef(cX, 0, 1, 0)  # on applique les rotations a la matrice du gl
        glRotatef(cY, 1, 0, 0)

        glMultMatrixf(modelMatrix)
        modelMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)  # on recharge l'ancienne matrice d'orientation
        glLoadIdentity()  # on reinitialise la matrice de position

        glTranslatef(0, 0, -20)
        glMultMatrixf(modelMatrix)
        DrawLowerCube(upFace)  # on dessine la partie inférieur du rubiks

        dtetha = 15

        if rotation_face == -1:
            if len(moove_buffer) == 0:
                if rotate == "LEFT":
                    rotation_face = upFace
                    angle = 5

                if rotate == "RIGHT":
                    rotation_face = upFace
                    angle = -5
            else:
                rotation_face = moove_buffer[0][0]
                angle = 5 * moove_buffer[0][1]

        else:
            angle += np.sign(angle) * np.sin((np.sign(angle) + 30) * np.pi / 180) * dtetha
            if np.abs(angle) >= 90:
                ColorFaceSwap(rotation_face, np.sign(angle))
                angle = 0
                rotation_face_animation(rotation_face, angle)
                rotation_face = -1
                if len(moove_buffer) > 0: moove_buffer = moove_buffer[1:]

                if np.array_equal(rubiksColor, rubiksColor_complete):
                    print("Congratulation !")
            else:
                rotation_face_animation(rotation_face, angle)
        modelMatrix2 = glGetFloatv(GL_MODELVIEW_MATRIX)
        glLoadIdentity()  # on reinitialise la matrice de position

        glMultMatrixf(modelMatrix2)
        DrawUpperCube(upFace)  # on dessine la partie supérieur du rubiks

        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(20)
        clock.tick()
        print(int(clock.get_fps()), "FPS")


Main()

# Grand merci mev pour l'idée

