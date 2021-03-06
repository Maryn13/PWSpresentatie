import numpy as np
import pygame
import math as math

class PO:

   x = 0
   #xpos
   y = 0 #y pos
   rot = 0 #radialen
   angvel = 0

   angacc = 0

   loc = np.array([800, 800])

   vel = np.array([0, 0])

   acc = np.array([0, 0])
   velscale = 0

   def __init__(self, name, width, length, weight, mps, cm, m, loc = None):

       self.name = name
       self.width = width
       self.length = length
       self.weight = weight
       self.mps = mps
       self.cm = cm
       self.m = m
       self.i = 2 / 3 * self.m * width
       if loc == None:
           self.loc = [np.random.randint(0, 1920), np.random.randint(0, 1080)]
       else:
           self.loc = loc
       self.dt = 1/60






   def applyForce(self, force_left, force_right):

       if (force_left >= 0 and force_right >= 0) or (force_left < 0 and force_right < 0):
           force_front = force_left + force_right - abs(force_left - force_right)
       elif (force_left < 0 and force_right == 0) or (force_right < 0 and force_left == 0):
           force_front = 0
       else:
           force_front = force_left + force_right  # + abs(force_left - force_right)

       force_rot = force_left - force_right
       force = np.array([math.cos(self.rot) * force_front, math.sin(self.rot) * force_front])

       fmag = math.sqrt(force[0]*force[0]+force[1]*force[1])
       alpha = 0
       if(force[0]!=0):
           alpha = math.atan(force[1]/force[0])
       else:
           if(force[1]>0):
               alpha = 0.5*3.141592653589793
           else:
               alpha = 1.5*3.141592653589793
       if(force[0]< 0):
           facc = np.array([math.cos(alpha+3.141592653589793) * fmag / self.m, math.sin(alpha+3.141592653589793) * fmag / self.m])
       else:
           facc = np.array([math.cos(alpha)*fmag/self.m, math.sin(alpha)*fmag/self.m])

       a = self.vel[0]
       b = self.vel[1]


       mag = math.sqrt(a * a + b * b)
       fwmag = mag *mag* 0.5 * 1.293 * 0.4 * 0.025 *1

       phi = 0
       if(self.vel[0]!= 0):
           phi = math.atan((self.vel[1]/self.vel[0]))

       if(a<0):
           fwacc = np.array([math.cos(phi+3.141592653589793)*fwmag/self.m, math.sin(phi+3.141592653589793)*fwmag/self.m])
       else:
           fwacc = np.array([math.cos(phi) * fwmag / self.m, math.sin(phi) * fwmag / self.m])
       facc = np.add(facc, -fwacc)
       self.acc = facc
       self.angacc = force_rot / self.i


   def action(self, choice):
       if choice == 0:
           self.applyForce(0,0)
       elif choice == 1:
           self.applyForce(75, 0)
       elif choice == 2:
           self.applyForce(0.0, 75)
       elif choice == 3:
           self.applyForce(75, 75)


       elif choice == 4:
           self.applyForce(75, 150)
       elif choice == 5:
           self.applyForce(150, 75)

       elif choice == 6:
           self.applyForce(-75, -75)
       elif choice == 7:
           self.applyForce(-150, -75)

       elif choice == 8:
           self.applyForce(-75, -150)

   def actionv(self, choice):
       if choice == 0:
           self.velscale = 0
           self.angvel = 0
       elif choice == 1:
           self.velscale = 50
       elif choice == 2:
           self.velscale = -50
       elif choice == 3:
           self.angvel = -0.1
       elif choice == 4:
           self.angvel = 0.1


   def setVelocityWithV(self):
       self.vel = np.array([math.cos(self.rot)*self.velscale, math.sin(self.rot)*self.velscale])


   def setVelocity(self):
       self.vel = np.add(self.vel, self.acc*self.dt)
       self.angvel += self.angacc*self.dt
       if self.angvel > 2.5:
           self.angvel = 2.5
       elif self.angvel < -2.5:
           self.angvel = -2.5






   def move(self):
       self.loc = np.add(self.loc, self.vel*self.dt)
       self.rot += self.angvel*self.dt
       self.rot = self.rot%(2*3.14159265358979323)

   def boundries(self, width, height):
       if self.loc[0] > width:
           self.loc[0] = width
       if self.loc[0]< 0:
           self.loc[0] = 0
       if self.loc[1] > height:
           self.loc[1] = height
       if self.loc[1]< 0:
           self.loc[1] = 0

   def draw(self, screen):
       p1 = np.array([0.5*self.width, 0.5*self.length])
       p2 = np.array([-0.5*self.width, 0.5*self.length])
       p3 = np.array([-0.5*self.width, -0.5*self.length])
       p4 = np.array([0.5*self.width, -0.5*self.length])
       modrot = np.array([[math.cos(self.rot), -math.sin(self.rot)],
                         [math.sin(self.rot), math.cos(self.rot)]])

       r = math.sqrt((0.5*self.width*0.5*self.width)+(0.5*self.width*0.5*self.width))

       pygame.draw.polygon(screen, [250, 0, 0], [tuple(np.add(self.loc, modrot.dot(p1))), tuple(np.add(self.loc, modrot.dot(p2))), tuple(np.add(self.loc, modrot.dot(p3))), tuple(np.add(self.loc, modrot.dot(p4)))])
