from . import BaseController
from tinyphysics import TinyPhysicsModel, STEER_RANGE, DEL_T, LAT_ACCEL_COST_MULTIPLIER, CONTEXT_LENGTH, MAX_ACC_DELTA, CONTROL_START_IDX
import numpy as np
import cvxpy as cp
from collections import namedtuple
from scipy.optimize import minimize

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
        self.driving_model = TinyPhysicsModel(model_path= "./models/tinyphysics.onnx", debug=False)
        # Add constraints for steering command within range
        self.prev_action = 0.0
        self.state_history = []
        self.action_history = []
        self.lataccel_history = []
        self.prev_optimum_cost = 0.0

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

    def calc_cost(self, u_init,target_lataccel, current_lataccel, state, future_plan,N):
        state_history = self.state_history
        lataccel_history = self.lataccel_history
        action_history = self.action_history
        for t in range(N):
                action_history.append(u_init[0])
                pred_state = self.driving_model.get_current_lataccel(
                    sim_states=state_history[-CONTEXT_LENGTH:],
                    actions=action_history[-CONTEXT_LENGTH:],
                    past_preds=lataccel_history[-CONTEXT_LENGTH:]
                    )
                #print("Next_state = ",pred_state)
                pred_state = np.clip(pred_state, current_lataccel - MAX_ACC_DELTA, current_lataccel + MAX_ACC_DELTA)
                lataccel_history.append(pred_state)
                current_lataccel = pred_state
                #update state history

                rolllataccel = future_plan.roll_lataccel[t]
                vego = future_plan.v_ego[t]
                aego = future_plan.a_ego[t]

                state_history.append(State(roll_lataccel=rolllataccel, v_ego=vego, a_ego=aego))

        lat_accel_cost = np.mean((np.array(future_plan.lataccel[:N]) - np.array(lataccel_history[-N:])) ** 2) * 100
        #print("Lateral accelaration cost:", lat_accel_cost)
        jerk_cost = np.mean((np.diff(np.array(lataccel_history[-N:])) / DEL_T) ** 2) * 100
        #print("Jerk cost:", jerk_cost)
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        print("Total Cost:",total_cost)

        return total_cost


    def update(self, target_lataccel, current_lataccel, state, future_plan):
        N = min(10, len(future_plan.lataccel))
        def minimize_wrapper(u_init):
            return self.calc_cost(u_init,target_lataccel, current_lataccel, state, future_plan,N)
        
        self.state_history.append(state)
        self.lataccel_history.append(current_lataccel)

        if N == 0 or N == 1:
            return self.prev_action
        
        if len(self.state_history) > CONTEXT_LENGTH:
            bounds = [(STEER_RANGE[0], STEER_RANGE[1])]*10

            u_initial = self.get_u_init(target_lataccel, current_lataccel, state, future_plan, N)
            result = minimize(minimize_wrapper,u_initial, bounds=bounds, method='SLSQP')
            print("Before optimization:", u_initial)
            print("Optimal cost:", result.fun)
            print("Optimal inputs:", result.x)
            #print("Status = ",problem.status)
            #print("Total_cost before",total_cost_before)
            #print("Optimal cost = ", opt_cost)
            #print("States:", [x[t].value for t in range(N)])
            #print("All values", steer_command.value)
            if result.fun > self.prev_optimum_cost:
                u_optimal = u_initial[0]
            else:
                u_optimal = result.x[0]
                self.prev_optimum_cost = result.fun
            self.action_history.append(u_optimal)
            self.prev_action = u_optimal
            return u_optimal
        
        else:
            #print("PID")
            error = (target_lataccel - current_lataccel)
            self.error_integral += error
            error_diff = error - self.prev_error
            self.prev_error = error
            action = self.p * error + self.i*self.error_integral + self.d*error_diff
            self.action_history.append(action)
            self.prev_action = action

            return action
