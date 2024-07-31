from . import BaseController
import numpy as np
from scipy.optimize import minimize
from tinyphysics import TinyPhysicsModel, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER, CONTEXT_LENGTH, MAX_ACC_DELTA, CONTROL_START_IDX
from collections import namedtuple
import pyswarms as ps

# Define the named tuples
State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

class Controller(BaseController):

    def __init__(self):
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0
        self.driving_model = TinyPhysicsModel(model_path="./models/tinyphysics.onnx", debug=False)
        self.prev_action = 0.0
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
    
    def get_u_init(self, target_lataccel, current_lataccel, state, future_plan, N):
        u = []
        error_cumulative = self.error_integral
        prev_error = self.prev_error
        prev_action =self.prev_action
        state_history = self.state_history
        action_history = self.action_history
        lataccel_history = self.lataccel_history

        for i in range(N):
            error = (target_lataccel - current_lataccel)
            error_cumulative += error
            error_diff = error - prev_error
            prev_error = error
            action = self.p * error + self.i*error_cumulative + self.d*error_diff
            action = np.clip(action, STEER_RANGE[0], STEER_RANGE[1])
            u.append(action)
            action_history.append(action)
            pred = self.driving_model.get_current_lataccel(
                    sim_states=state_history[-CONTEXT_LENGTH:],
                    actions=action_history[-CONTEXT_LENGTH:],
                    past_preds=lataccel_history[-CONTEXT_LENGTH:]
                    )
            pred = np.clip(pred, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
            lataccel_history.append(pred)
            current_lataccel = pred
            target_lataccel = future_plan.lataccel[i]
            rolllataccel = future_plan.roll_lataccel[i]
            vego = future_plan.v_ego[i]
            aego = future_plan.a_ego[i]

            state_history.append(State(roll_lataccel=rolllataccel, v_ego=vego, a_ego=aego))
        
        return u

    def cost_function(self, u,current_lataccel, future_plan, state_history, target_lataccel,action_history,lataccel_history):
        
        n_particles = u.shape[0]
        N = u.shape[1]
        total_costs = np.zeros(n_particles)
        
        for p in range(n_particles):
            dummy_action_history = action_history.copy()
            particle_cost = 0.0
            particle_lataccel_history = lataccel_history.copy()
            # Compute next states and costs
            for t in range(N):
                dummy_action_history.append(u[p,t])
                #print(dummy_action_history)
                # Predict next state using the driving model
                pred_state = self.driving_model.get_current_lataccel(
                    sim_states=state_history[-CONTEXT_LENGTH:],
                    actions=dummy_action_history[-CONTEXT_LENGTH:],
                    past_preds=particle_lataccel_history[-CONTEXT_LENGTH:]
                )
                pred_state = np.clip(pred_state, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
                particle_lataccel_history.append(pred_state)
                
                # Calculate costs
                if t > 0:
                    jerk_cost = (((pred_state - lataccel_history[-2])/DEL_T)**2)*100
                    lataccel_cost = ((pred_state - future_plan.lataccel[t-1])**2)*100
                else:
                    lataccel_cost = ((pred_state - target_lataccel)**2)*100
                    jerk_cost = (((pred_state - current_lataccel)/DEL_T)**2)*100
                particle_cost += lataccel_cost*LAT_ACCEL_COST_MULTIPLIER + jerk_cost
            
            total_costs[p] = particle_cost
        #print(total_cost)
        return total_costs

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        self.state_history.append(state)
        self.lataccel_history.append(current_lataccel)

        state_history = self.state_history
        lataccel_history = self.lataccel_history
        action_history = self.action_history
        N = min(len(future_plan.lataccel),2)
        if N == 0 or N == 1:
            return self.prev_action
        
        if len(self.state_history) > CONTROL_START_IDX:
            u_initial = self.get_u_init(target_lataccel, current_lataccel, state, future_plan, N)
            print(u_initial)
            # Define the bounds for the control actions
            bounds = (np.full(len(u_initial), STEER_RANGE[0]), np.full(len(u_initial), STEER_RANGE[1]))
            init_pos = np.tile(u_initial, (5, 1))

            # Set up PSO optimizer
            optimizer = ps.single.GlobalBestPSO(
                n_particles=5, 
                dimensions=len(u_initial),
                options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                bounds=bounds,
                init_pos= init_pos
            )

            def wrapped_cost_function(u):
                return self.cost_function(u,current_lataccel, future_plan, state_history, target_lataccel,action_history,lataccel_history)

            # Run optimization
            cost, pos = optimizer.optimize(
                wrapped_cost_function, 
                iters=500, 
            )
            
            print("Optimal control actions:", pos)
            print("Optimal cost:", cost)
            
            self.action_history.append(pos[0])
            self.prev_action = pos[0]
            return pos[0]
        
        else:
            error = (target_lataccel - current_lataccel)
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            action = self.p * error + self.i * self.error_integral + self.d * error_diff
            self.action_history.append(action)
            self.prev_action = action
            return action
