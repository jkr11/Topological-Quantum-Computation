import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2
phi_inv = phi**(-1)

B1 = np.array([
    [np.exp(-4j * np.pi / 5), 0],
    [0, np.exp(3j * np.pi / 5)]
])

B2 = np.array([
    [
        phi**(-1) * np.exp(4j * np.pi / 5),
        phi**(-1 / 2) * np.exp(-3j * np.pi / 5)
    ],
    [phi**(-1 / 2) * np.exp(-3j * np.pi / 5), -1 * phi**(-1)]
])

B1_inv = np.linalg.inv(B1)
B2_inv = np.linalg.inv(B2)

gate_set = [B1, B2, B1_inv, B2_inv]

def brute_force_search(U_target, M, depth, state, min, min_state):
  """Brute forces braids of length depth over gate_set"""
  if (depth == 0):
    res = norm(U_target, M)
    if (res < min):
      return res, state

    return min, min_state

  for i, g in enumerate(gate_set):
    ns = state.copy()
    if i + 2 or i - 2 in ns:
      pass
    ns.append(i)
    min, min_state = brute_force_search(U_target, M @ g, depth - 1, ns, min,
                                        min_state)

  return min, min_state


def eval_braid(braid):
  """evaluates the unitary representation of the braids (left multiplied)"""
  res = gate_set[braid[0]]
  for i in range(1, len(braid)):
    res = gate_set[braid[i]] @ res
  return res


def square_mod(x):
  return x.real**2 + x.imag**2


def norm(U, V):
  """Schatten norm from Bon+05"""
  D = U.shape[0]
  U_t = np.conj(np.transpose(U))
  UtV = np.dot(U_t, V)
  trace = np.trace(UtV)
  distance = np.sqrt(1 - (np.abs(trace) / D))
  return distance


IMAGES = ["sigma1.png","sigma2.png","sigma1_inv.png", "sigma2_inv.png"]


def concat(img1, img2):
  """concatenates two images of matching shapes"""
  h1, w1 = img1.shape[:2]
  h2, w2 = img2.shape[:2]
  maxh = max(h1, h2)
  w = w1 + w2
  new_img = np.zeros(shape=(maxh, w, 3))
  new_img[:h1, :w1] = img1
  new_img[:h2, w1:w1 + w2] = img2
  return new_img


def concatn(list):
  """conatenates n images from the left"""
  output = None
  for i, img in enumerate(list):
    img = plt.imread(img)[:, :, :3]
    if i == 0:
      output = img
    else:
      output = concat(output, img)
  return output


def draw(braid, fn=None):
  """draws the braid symbolically"""
  images = []
  for b in (braid):
    images += [IMAGES[b]]
  print(images)
  output = concatn(images)
  plt.imshow(output)
  name = ""
  for i in braid:
    if i in [0,1]:
      name += f"$\sigma_{i+1}$"
    elif i in [2,3]:
      name += f"$\sigma_{i-1}^{-1}$"
  plt.title(name)
  plt.grid(False)
  plt.axis('off')
  plt.show()


U_target1 = np.array([[0, 1], [1, 0]])
U_target2 = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
U_target3 = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

min, state = brute_force_search(U_target3, np.eye(2), 9, [], np.inf, [])

print(min, state)
draw(state)
print(eval_braid(state))

