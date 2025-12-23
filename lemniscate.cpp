#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_angular_velocity.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_thrust_setpoint.hpp>
#include <px4_msgs/msg/vehicle_torque_setpoint.hpp>
#include <px4_msgs/msg/hover_thrust_estimate.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/actuator_motors.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
// #include <px4_ros_com/frame_transforms.h>
#include <Eigen/Dense>

#include "../include/gazebo_ai.h"

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;

struct PIDGains {
    float kp, ki, kd;
};

struct LQRParams {
    double Q_pos_xy;
    double Q_pos_z;
    double Q_vel_xy;
    double Q_vel_z;
    double Q_ang_xy;
    double Q_ang_z;
    double Q_rate_xy;
    double Q_rate_z;

    double R_thrust;
    double R_torque_xy;
    double R_torque_z;
};




class OffboardControl : public rclcpp::Node
{
public:
    OffboardControl() : Node("offboard_control")
    {

        // Hardcoded because the calculation of pesudo-inverse is not accurate
        throttles_to_normalized_torques_and_thrust_ <<
                -0.5718,    0.4376,    0.5718,   -0.4376,
                -0.3536,    0.3536,   -0.3536,    0.3536,
                -0.2832,   -0.2832,  0.2832,  0.2832,
                0.2500,   0.2500,   0.2500,   0.2500;
        //load pid gains from yaml
        std::vector<std::string> axes = {"pid_x", "pid_y", "pid_z", "ori_xyz"};

        for (const auto &axis : axes) {
            PIDGains gains;

            this->declare_parameter(axis + ".kp", 0.0);
            this->declare_parameter(axis + ".ki", 0.0);
            this->declare_parameter(axis + ".kd", 0.0);

            this->get_parameter(axis + ".kp", gains.kp);
            this->get_parameter(axis + ".ki", gains.ki);
            this->get_parameter(axis + ".kd", gains.kd);

            pid_map_[axis] = gains;

            RCLCPP_INFO(this->get_logger(), "Loaded %s gains: Kp=%.2f, Ki=%.2f, Kd=%.2f",
                        axis.c_str(), gains.kp, gains.ki, gains.kd);
        }

        for (int i = 0; i < 11; i++)
        {
            this->declare_parameter("coefs." + std::to_string(i), 0.0);
            this->get_parameter("coefs." + std::to_string(i), coefs[i]);
        }

        // Load LQR parameters
        this->declare_parameter("lqr.Q_pos_xy", 10.0);
        this->declare_parameter("lqr.Q_pos_z", 20.0);
        this->declare_parameter("lqr.Q_vel_xy", 1.0);
        this->declare_parameter("lqr.Q_vel_z", 1.0);
        this->declare_parameter("lqr.Q_ang_xy", 5.0);
        this->declare_parameter("lqr.Q_ang_z", 1.0);
        this->declare_parameter("lqr.Q_rate_xy", 0.1);
        this->declare_parameter("lqr.Q_rate_z", 0.1);

        this->declare_parameter("lqr.R_thrust", 0.1);
        this->declare_parameter("lqr.R_torque_xy", 1.0);
        this->declare_parameter("lqr.R_torque_z", 1.0);

        this->get_parameter("lqr.Q_pos_xy", lqr_params.Q_pos_xy);
        this->get_parameter("lqr.Q_pos_z", lqr_params.Q_pos_z);
        this->get_parameter("lqr.Q_vel_xy", lqr_params.Q_vel_xy);
        this->get_parameter("lqr.Q_vel_z", lqr_params.Q_vel_z);
        this->get_parameter("lqr.Q_ang_xy", lqr_params.Q_ang_xy);
        this->get_parameter("lqr.Q_ang_z", lqr_params.Q_ang_z);
        this->get_parameter("lqr.Q_rate_xy", lqr_params.Q_rate_xy);
        this->get_parameter("lqr.Q_rate_z", lqr_params.Q_rate_z);

        this->get_parameter("lqr.R_thrust", lqr_params.R_thrust);
        this->get_parameter("lqr.R_torque_xy", lqr_params.R_torque_xy);
        this->get_parameter("lqr.R_torque_z", lqr_params.R_torque_z);

        computeLQRGains();

        //print generated pid_map_ and values
        std::cout << "PID gains loaded: " << std::endl;
        for (const auto &axis : axes) {
            std::cout << axis << ": " << pid_map_[axis].kp << ", " << pid_map_[axis].ki << ", " << pid_map_[axis].kd << std::endl;
        }

        std::cout << "Generated control system coefs: " << coefs.transpose() << std::endl;

        precomputeTrajectory(0.01f);
        std::cout << "Precomputed trajectory of size: " << traj_.size() << std::endl;

        // pubs
        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>("/fmu/in/offboard_control_mode", 10);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>("/fmu/in/vehicle_command", 10);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>("/fmu/in/trajectory_setpoint", 10);
        thrust_setpoint_publisher_ = this->create_publisher<px4_msgs::msg::VehicleThrustSetpoint>("/fmu/in/vehicle_thrust_setpoint", 10);
        torque_setpoint_publisher_ = this->create_publisher<px4_msgs::msg::VehicleTorqueSetpoint>("/fmu/in/vehicle_torque_setpoint", 10);
        rates_setpoint_publisher_ = this->create_publisher<px4_msgs::msg::VehicleRatesSetpoint>("/fmu/in/vehicle_rates_setpoint", 10);

        // subs
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);


        vehicle_local_position_subscriber_ = this->create_subscription<VehicleLocalPosition>("/fmu/out/vehicle_local_position", qos, [this](px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
        {
            Eigen::Vector<float, 3> p_W = rotateVectorFromToENU_NED(Eigen::Vector<float, 3>(msg->x, msg->y, msg->z));
            local_position_x_ = p_W[0];
            local_position_y_ = p_W[1];
            local_position_z_ = p_W[2];

            Eigen::Vector3f v_W = rotateVectorFromToENU_NED(Eigen::Vector<float, 3>(msg->vx, msg->vy, msg->vz));

            local_velocity_x_ = v_W[0];
            local_velocity_y_ = v_W[1];
            local_velocity_z_ = v_W[2];
        });

        vehicle_attitude_subscriber_ = this->create_subscription<VehicleAttitude>("/fmu/out/vehicle_attitude", qos, [this](px4_msgs::msg::VehicleAttitude::SharedPtr msg)
        {
            Eigen::Quaternionf orientation_B_W(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
            orientation_B_W = rotateQuaternionFromToENU_NED(orientation_B_W);
            Eigen::Vector3f euler = getEulerFromRotationMatrix(orientation_B_W.toRotationMatrix());
            local_ori_x_ = euler[0];
            local_ori_y_ = euler[1];
            local_ori_z_ = euler[2];
        });

        vehicle_angular_velocity_subscriber_ = this->create_subscription<VehicleAngularVelocity>("/fmu/out/vehicle_angular_velocity", qos, [this](px4_msgs::msg::VehicleAngularVelocity::SharedPtr msg)
        {
            Eigen::Vector<float, 3> av_B = rotateVectorFromToFRD_FLU(Eigen::Vector<float, 3>(msg->xyz[0], msg->xyz[1], msg->xyz[2]));

            local_angular_velocity_x_ = av_B[0];
            local_angular_velocity_y_ = av_B[1];
            local_angular_velocity_z_ = av_B[2];
        });

        vehicle_status_subscriber_ = this->create_subscription<VehicleStatus>(
                                         "/fmu/out/vehicle_status_v1", qos, [this](VehicleStatus::UniquePtr msg)
        {
            if (msg->nav_state_user_intention == 14)
            {
                radio_controller_sending_offboard = true;
            }
            else
            {
                radio_controller_sending_offboard = false;
            }
        });

        hover_thrust_setpoint_subscriber_ = this->create_subscription<HoverThrustEstimate>(
                                                "/fmu/out/hover_thrust_estimate", qos, [this](HoverThrustEstimate::UniquePtr msg)
        {
            hte = msg->hover_thrust;
        });

        actuator_motors_subscriber_ = this->create_subscription<ActuatorMotors>(
                                          "/fmu/out/actuator_motors", qos, [this](ActuatorMotors::UniquePtr msg)
        {
            motor_0 = msg->control[0];
            motor_1 = msg->control[1];
            motor_2 = msg->control[2];
            motor_3 = msg->control[3];
        });


        // PX4 requires an initial setpoint before entering offboard mode
        px4_msgs::msg::TrajectorySetpoint setpoint;
        setpoint.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        setpoint.position = {0.0, 0.0, -1.0};
        setpoint.velocity = {NAN, NAN, NAN};
        setpoint.acceleration = {NAN, NAN, NAN};
        setpoint.jerk = {NAN, NAN, NAN};
        setpoint.yaw = NAN;
        setpoint.yawspeed = NAN;
        trajectory_setpoint_publisher_->publish(setpoint);


        // log file
        traj_log_.open("traj.csv");
        traj_log_ << "x,y,z,gx,gy,gz,motor0,motor1,motor2,motor3,target_yaw,yaw" << std::endl;

        // Runner
        auto timer_callback = [this]() -> void
        {
            if(radio_controller_sending_offboard)
            {
                if (offboard_setpoint_counter_ == 10)
                {
                    // Change to Offboard mode after 10 setpoints
                    this->publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);

                    // Arm the vehicle
                    this->arm();
                }

                // // offboard_control_mode needs to be paired with trajectory_setpoint
                publish_offboard_control_mode();

                state_manager();

                // stop the counter after reaching 11
                if (offboard_setpoint_counter_ < 11)
                {
                    offboard_setpoint_counter_++;
                }
            }
            else
            {
                // PX4 uses this message as a heartbeat
                publish_offboard_control_mode();
            }
        };

        timer_ = this->create_wall_timer(8ms, timer_callback); // just call as fast as possible. check dt in functions.
    }

    ~OffboardControl() // Destructor
    {
        if (traj_log_.is_open())
            traj_log_.close();
    }

    inline Eigen::Vector<float,3> rotateVectorFromToENU_NED(const Eigen::Vector<float,3>& vec_in) {
        // NED (X North, Y East, Z Down) & ENU (X East, Y North, Z Up)
        Eigen::Vector<float,3> vec_out;
        vec_out << vec_in[1], vec_in[0], -vec_in[2];
        return vec_out;
    }

    inline Eigen::Vector<float,3> rotateVectorFromToFRD_FLU(const Eigen::Vector<float,3>& vec_in) {
        // FRD (X Forward, Y Right, Z Down) & FLU (X Forward, Y Left, Z Up)
        Eigen::Vector<float,3> vec_out;
        vec_out << vec_in[0], -vec_in[1], -vec_in[2];
        return vec_out;
    }

    inline Eigen::Vector3f getEulerFromRotationMatrix(const Eigen::Matrix<float, 3, 3> &R) {
        float s = -R(2,0);
        if (s > 1.0f)  s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        float pitch = std::asin(s);
        float roll, yaw;
        if (std::fabs(std::cos(pitch)) > 1e-6) {
            roll = std::atan2(R(2,1), R(2,2));
            yaw  = std::atan2(R(1,0), R(0,0));
        } else {
            roll = 0;
            yaw = std::atan2(-R(0,1), R(1,1));
        }
        return Eigen::Vector3f(roll, pitch, yaw);
    }

    inline Eigen::Quaternionf rotateQuaternionFromToENU_NED(const Eigen::Quaternionf& quat_in) {
        // Transform from orientation represented in ROS format to PX4 format and back
        //  * Two steps conversion:
        //  * 1. aircraft to NED is converted to aircraft to ENU (NED_to_ENU conversion)
        //  * 2. aircraft to ENU is converted to baselink to ENU (baselink_to_aircraft conversion)
        // OR
        //  * 1. baselink to ENU is converted to baselink to NED (ENU_to_NED conversion)
        //  * 2. baselink to NED is converted to aircraft to NED (aircraft_to_baselink conversion
        // NED_ENU_Q Static quaternion needed for rotating between ENU and NED frames
        Eigen::Vector<float, 3> euler_1(M_PI, 0.0, M_PI_2);
        Eigen::Quaternionf NED_ENU_Q(Eigen::AngleAxisf(euler_1.z(), Eigen::Vector3f::UnitZ()) *
                                     Eigen::AngleAxisf(euler_1.y(), Eigen::Vector3f::UnitY()) *
                                     Eigen::AngleAxisf(euler_1.x(), Eigen::Vector3f::UnitX()));

        // AIRCRAFT_BASELINK_Q Static quaternion needed for rotating between aircraft and base_link frames
        Eigen::Vector<float, 3> euler_2(M_PI, 0.0, 0.0);
        Eigen::Quaternionf AIRCRAFT_BASELINK_Q(Eigen::AngleAxisf(euler_2.z(), Eigen::Vector3f::UnitZ()) *
                                               Eigen::AngleAxisf(euler_2.y(), Eigen::Vector3f::UnitY()) *
                                               Eigen::AngleAxisf(euler_2.x(), Eigen::Vector3f::UnitX()));

        return (NED_ENU_Q*quat_in)*AIRCRAFT_BASELINK_Q;
    }

    inline void eigenOdometryFromPX4Msg(px4_msgs::msg::VehicleOdometry::SharedPtr msg,
                                        Eigen::Vector<float, 3>& position_W, Eigen::Quaternionf& orientation_B_W,
                                        Eigen::Vector<float, 3>& velocity_B, Eigen::Vector<float, 3>& angular_velocity_B) {

        position_W = rotateVectorFromToENU_NED(Eigen::Vector<float, 3>(msg->position[0], msg->position[1], msg->position[2]));

        Eigen::Quaternionf quaternion(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
        orientation_B_W = rotateQuaternionFromToENU_NED(quaternion);

        velocity_B = rotateVectorFromToENU_NED(Eigen::Vector<float, 3>(msg->velocity[0], msg->velocity[1], msg->velocity[2]));

        angular_velocity_B = rotateVectorFromToFRD_FLU(Eigen::Vector<float, 3>(msg->angular_velocity[0], msg->angular_velocity[1], msg->angular_velocity[2]));
    }

    inline Eigen::Matrix3f euler_to_rotation(float roll, float pitch, float yaw) {
        Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());

        // Multiplying AngleAxis returns a Quaternion, so cast to matrix
        Eigen::Matrix3f R = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
        return R;
    }


    inline Eigen::Vector3f rotation_matrix_to_axis_angle(const Eigen::Matrix3f& R_err) {
        Eigen::AngleAxisf angleAxis(R_err);
        Eigen::Vector3f axis = angleAxis.axis();
        float angle = angleAxis.angle();
        return axis * angle;  // Rotation vector (axis * angle)
    }


    Eigen::Vector<float, 3> model(float collective_thrust, float dt);

private:
    bool radio_controller_sending_offboard = false;
    rclcpp::TimerBase::SharedPtr timer_;
    // Publishers
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    // rclcpp::Publisher<VehicleRatesSetpoint>::SharedPtr rates_setpoint_publisher_;
    rclcpp::Publisher<px4_msgs::msg::VehicleThrustSetpoint>::SharedPtr thrust_setpoint_publisher_;
    rclcpp::Publisher<px4_msgs::msg::VehicleTorqueSetpoint>::SharedPtr torque_setpoint_publisher_;
    rclcpp::Publisher<px4_msgs::msg::VehicleRatesSetpoint>::SharedPtr rates_setpoint_publisher_;
    std::vector<Eigen::Vector4f> traj_;

    // Subscribers
    rclcpp::Subscription<VehicleStatus>::SharedPtr vehicle_status_subscriber_;

    rclcpp::Subscription<VehicleLocalPosition>::SharedPtr vehicle_local_position_subscriber_;
    rclcpp::Subscription<VehicleAttitude>::SharedPtr vehicle_attitude_subscriber_;
    rclcpp::Subscription<VehicleAngularVelocity>::SharedPtr vehicle_angular_velocity_subscriber_;
    rclcpp::Subscription<ActuatorMotors>::SharedPtr actuator_motors_subscriber_;

    rclcpp::Subscription<HoverThrustEstimate>::SharedPtr hover_thrust_setpoint_subscriber_;

    std::unordered_map<std::string, PIDGains> pid_map_;
    LQRParams lqr_params;
    std::ofstream traj_log_;
    int t = 0;


    Eigen::Vector<float, 11> coefs = Eigen::Vector<float, 11>::Zero();

    std::chrono::steady_clock::time_point start_time_;
    // Quad state
    float local_position_x_ = 0;
    float local_position_y_ = 0;
    float local_position_z_ = 0;
    float local_velocity_x_ = 0;
    float local_velocity_y_ = 0;
    float local_velocity_z_ = 0;
    bool second_loop = false;
    NeuralNetwork nn;
    float motor_0 = 0;
    float motor_1 = 0;
    float motor_2 = 0;
    float motor_3 = 0;

    //hover thrust estimate storage
    float hte = 0.7;

    //note, stored for distrubance observer
    float prev_local_velocity_x_ = 0;
    float prev_local_velocity_y_ = 0;
    float prev_local_velocity_z_ = 0;
    float prev_collective_thrust_ = 0;
    Eigen::Quaternionf prev_ori_q;

    Eigen::Quaternionf ori_q;
    float local_angular_velocity_x_ = 0;
    float local_angular_velocity_y_ = 0;
    float local_angular_velocity_z_ = 0;
    float local_ori_x_ = 0;
    float local_ori_y_ = 0;
    float local_ori_z_ = 0;
    float dx_prev = 0;
    float dy_prev = 0;
    float dz_prev = 0;
    int ascended_counter_ = 0;

    float lv;
    float la;

    // flags used to control current state (see. state_manager())
    bool told_to_land = false;
    bool ascended_ = false;
    bool finished_ = false;
    bool is_first_run = true;
    int laps = 0;

    float t_z_ = .5f;

    std::atomic<uint64_t> timestamp_; //!< common synced timestamped

    uint64_t offboard_setpoint_counter_ = 0; //!< counter for the number of setpoints sent


    // pid stuff
    bool debug = false; // Set to false to disable debug output
    std::chrono::steady_clock::time_point prev_time{};
    float x_prev = 0.0;
    float y_prev = 0.0;
    float altitude_prev = 0.0;
    Eigen::Vector3f ori_prev = Eigen::Vector3f(0, 0, 0);
    float z_i = 0.0f;
    float x_i = 0.0f;
    float y_i = 0.0f;
    Eigen::Vector<float, 3> ori_i = Eigen::Vector<float, 3>(0, 0, 0);
    Eigen::Vector4f prev_action = Eigen::Vector4f(0, 0, 0, 0);

    // model stuff
    Eigen::Vector<float, 3> rotor_drag{4.5, 4.5, 7.0};
    Eigen::Vector<float, 3> prev_cmd_vel{0, 0, 0};
    float arm_l = 0.225;
    float mass = 2.0;
    Eigen::Matrix<float, 3, 3> J = mass / 12.0 * arm_l * arm_l * rotor_drag.asDiagonal(); // inertia matrix
    Eigen::Matrix<float, 3, 3> pGainBodyRates =
        Eigen::Vector<float, 3>(4.5, 4.5, 4.0).asDiagonal();

    Eigen::Vector<float, 3> L = {0.0002,0.0002,0.0002};
    Eigen::Vector<float, 3> d = {0,0,0};
    // thrust to moment
    static constexpr float rotor_drag_coef = 0.016;
    const Eigen::Matrix<float, 3, 4> t_BM_ = arm_l * std::sqrt(0.5) *
            (Eigen::Matrix<float, 3, 4>() <<
             1, -1, -1, 1,
             -1, -1, 1, 1,
             0, 0, 0, 0).finished(); // thrust to moment matrix

    const Eigen::Matrix<float, 4, 4> B_allocation_ = (Eigen::Matrix<float, 4, 4>() <<


            Eigen::Vector<float, 4>::Ones().transpose(),

            t_BM_.topRows<2>(),

            rotor_drag_coef * Eigen::Vector<float, 4>(1, -1, 1, -1).transpose()

                                                     )
            .finished(); // Used to generate thrust equations (see torque matrix eq. )
    const Eigen::Matrix<float, 4, 4> B_allocation_inv_ = B_allocation_.inverse();

    Eigen::Matrix<float, 4, 4> throttles_to_normalized_torques_and_thrust_ = Eigen::Matrix<float, 4, 4>::Zero();


    // RVOLMEA CONSTANTS
    float z_prev = 0.0;
    float x_integral = 0.0;
    float y_integral = 0.0;
    float z_integral = 0.0;

    float vx_smooth = 0.0;
    float vy_smooth = 0.0;
    float vz_smooth = 0.0;
    float ox_smooth = 0.0;
    float oy_smooth = 0.0;
    float oz_smooth = 0.0;
    int idx = 0;
    float eps = 1e-6;
    float fault_scale = 1.0;
    // RVOLMEA FUNCTIONS

    // RVOLMEA FUNCTIONS END

    void publish_offboard_control_mode();
    void state_manager();
    void check_radio_controller();
    void get_to_start_pos();
    void arm();
    void disarm();
    void land();
    void calculate_thrust_torque_setpoint();
    void publish_thrust_torque_setpoint(Eigen::Vector<float,4> setpoint);
    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0);
    void rlControl();
    Eigen::Vector4f getPositionAtTime(float t);
    void rvolmeaControl();
    void lqrControl();
    void computeLQRGains();
    void precomputeTrajectory(float dt);
    float prev_local_ori_x = 0.0f;

    Eigen::Matrix<float, 4, 12> K_lqr;
    bool k_lqr_initialized = false;
};


/// RVOLMEA GENERATED FUNCTIONS




//RVOLMEA FUNCTIONS END


/**
 * @brief Publish the thrust and torque setpoint to PX4.
 *
 * @param setpoint non-normalized thrust and torque setpoint of the form [thrust, torque_x, torque_y, torque_z] in FLU co-ordinate frame.
 */
void OffboardControl::publish_thrust_torque_setpoint(Eigen::Vector<float,4> setpoint)
{
    // Note, in PX4 thrust appears normalized relative to the sqrt of max motor RPM
    // to normalize properly, we need to convert thrust/torque to motor RPM
    // normalize the rpm, then convert back to thrust/torque.

    // std::cout << "Setpoint: " << setpoint << std::endl;




    constexpr float max_thrust = 27.0f;

    setpoint[0] = std::clamp(setpoint[0], 0.0f, max_thrust);
    // setpoint[1] = std::clamp(setpoint[1], -2.77f, 2.77f);
    // setpoint[2] = std::clamp(setpoint[2], -2.77f, 2.77f);
    // setpoint[3] = std::clamp(setpoint[3], -2.77f, 2.77f);

    // Eigen::Vector<float, 4> thrust_and_torque = B_allocation_inv_ * setpoint;
    // thrust_and_torque /= single_rotor_max_thurst;
    // Eigen::Vector<float, 4> omega = thrust_and_torque * 1000.0f; //convert to rpm
    // omega = omega.cwiseSqrt();


    // omega *= 1000.0f;

    // // Clamp motor thrust
    // Eigen::Vector<float, 4> motor_thrusts = omega.cwiseProduct(omega) * float(7.84e-6);

    // // Forces and moments
    // Eigen::Vector<float, 4> normalized_thrust_torque = B_allocation_ * motor_thrusts; // now we can do relative.
    Eigen::Vector<float, 4> normalized_thrust_torque = setpoint;
    normalized_thrust_torque[0] /= max_thrust;
    // normalized_thrust_torque[1] /= 2.77f; //calculated by putting max values intp B_alllocation table
    // normalized_thrust_torque[2] /= 2.77f;
    // normalized_thrust_torque[3] /= 2.77f;

    normalized_thrust_torque = normalized_thrust_torque.cwiseMin(1.0f).cwiseMax(-1.0f);

    px4_msgs::msg::VehicleThrustSetpoint thrust_sp_msg;
    px4_msgs::msg::VehicleTorqueSetpoint torque_sp_msg;
    thrust_sp_msg.timestamp_sample = this->get_clock()->now().nanoseconds() / 1000;
    torque_sp_msg.timestamp_sample = thrust_sp_msg.timestamp_sample;
    thrust_sp_msg.timestamp = thrust_sp_msg.timestamp_sample;
    torque_sp_msg.timestamp = thrust_sp_msg.timestamp_sample;
    // Fill thrust setpoint msg
    float normalized_thrust = std::clamp(normalized_thrust_torque[0], 0.0f, 1.0f);
    thrust_sp_msg.xyz[0] = 0.0;
    thrust_sp_msg.xyz[1] = 0.0;
    if (setpoint[0] > 0.1) {
        thrust_sp_msg.xyz[2] = -normalized_thrust;
    }
    else {
        thrust_sp_msg.xyz[2] = -0.1; // 10% min thrust
    }
    // Rotate torque setpoints from FLU to FRD and fill the msg
    Eigen::Vector3f torque_Nm(normalized_thrust_torque[1], normalized_thrust_torque[2], normalized_thrust_torque[3]);
    Eigen::Vector<float, 3> rotated_torque_sp = rotateVectorFromToFRD_FLU(torque_Nm);
    torque_sp_msg.xyz[0] = rotated_torque_sp[0];
    torque_sp_msg.xyz[1] = rotated_torque_sp[1];
    torque_sp_msg.xyz[2] = rotated_torque_sp[2];
    // Publish msgs
    // thrust_setpoint_publisher_->publish(thrust_sp_msg);
    // torque_setpoint_publisher_->publish(torque_sp_msg);

    px4_msgs::msg::VehicleRatesSetpoint rates_sp_msg;
    rates_sp_msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    rates_sp_msg.roll = rotated_torque_sp[0];
    rates_sp_msg.pitch = rotated_torque_sp[1];
    rates_sp_msg.yaw = rotated_torque_sp[2];
    rates_sp_msg.thrust_body[2] = thrust_sp_msg.xyz[2];
    rates_setpoint_publisher_->publish(rates_sp_msg);
    // std::cout << "Thrust sp: " << normalized_thrust << ", Torque sp: " << normalized_thrust_torque[1] << "," << normalized_thrust_torque[2] << "," << normalized_thrust_torque[3] << std::endl;
}

/**
 * @brief Send a command to Land the vehicle
 */
void OffboardControl::land()
{
    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_NAV_LAND);
    RCLCPP_INFO(this->get_logger(), "Land command send");
}

/**
 * @brief Send a command to Arm the vehicle
 */
void OffboardControl::arm()
{
    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);

    RCLCPP_INFO(this->get_logger(), "Arm command send");
}

/**
 * @brief Send a command to Disarm the vehicle
 */
void OffboardControl::disarm()
{
    publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0);

    RCLCPP_INFO(this->get_logger(), "Disarm command send");
}

void OffboardControl::rvolmeaControl()
{
    auto now = std::chrono::steady_clock::now();

    bool debug = false;

    if (prev_time.time_since_epoch().count() == 0) {
        if (debug) std::cout << "[DEBUG] First run, setting prev_time.\n";
        prev_time = now;
        return;
    }

    float dt = std::chrono::duration<float>(now - prev_time).count(); // (could be integrator dt)

    constexpr float max_thrust = 44.0f;

    if (debug) std::cout << "[DEBUG] Time delta (dt): " << dt << "\n";

    prev_time = now;



    Eigen::Vector3f position = Eigen::Vector3f(local_position_x_, local_position_y_, local_position_z_);
    Eigen::Vector4f target = getPositionAtTime(t*0.004f);
    // Eigen::Vector4f target(0.0f, 0.0f, 4.0f, 0.0f);
    Eigen::Vector3f goal_position = target.head(3);
    float target_yaw = target[3];
    Eigen::Vector3f linear_velocity = Eigen::Vector3f(local_velocity_x_, local_velocity_y_, local_velocity_z_);
    Eigen::Vector3f angular_velocity = Eigen::Vector3f(local_angular_velocity_x_, local_angular_velocity_y_, local_angular_velocity_z_);
    Eigen::Vector3f ypr = Eigen::Vector3f(local_ori_z_, local_ori_y_, local_ori_x_);
    traj_log_ <<  local_position_x_ << "," << local_position_y_ << "," << local_position_z_ << "," << target[0] << "," << target[1] << "," << target[2] << "," << motor_0 << "," << motor_1 << "," << motor_2 << "," << motor_3 << "," << target_yaw << "," << local_ori_z_ << std::endl;
    t++;

    Eigen::Vector<float, 24> coefs {-0.315995,3.59996,-0.393737,2.3025,0.0471963,1.55924,-7.8674,0.623645,0.309091,4.14766,0.108078,5.37875,1.08379,-0.131335,6.69693,5.62777,2.25677,1.11582,6.50565,0.839767,7.43786,5.99279,5.15347,0.0340794};


    const float g = 9.81f;
    const Eigen::Vector3f gravity_vec(0.0f,0.0f,g);

    // Extract positional errors
    const float x_err = goal_position[0] - position[0];
    const float y_err = goal_position[1] - position[1];
    const float z_err = goal_position[2] - position[2];

    const float vx = linear_velocity[0];
    const float vy = linear_velocity[1];
    const float vz = linear_velocity[2];
    const float ox = angular_velocity[0];
    const float oy = angular_velocity[1];
    const float oz = angular_velocity[2];


    const float beta = std::clamp(coefs[0], 0.7f, 0.990f);

    vx_smooth = beta * vx_smooth + (1.0f - beta) * vx;
    vy_smooth = beta * vy_smooth + (1.0f - beta) * vy;
    vz_smooth = beta * vz_smooth + (1.0f - beta) * vz;

    ox_smooth = beta * ox_smooth + (1.0f - beta) * ox;
    oy_smooth = beta * oy_smooth + (1.0f - beta) * oy;
    oz_smooth = beta * oz_smooth + (1.0f - beta) * oz;

    const float vel_mag = std::sqrt(vx_smooth*vx_smooth + vy_smooth*vy_smooth + vz_smooth*vz_smooth);

    const float dyn_gain = 1.0f + coefs[4] * vel_mag;

    auto clamp_integral = [](float integral, float min_val, float max_val) {
        if (integral > max_val) return max_val;
        if (integral < min_val) return min_val;
        return integral;
    };

    // Error derivatives
    const float dz = (z_err - z_prev) / dt;
    const float dx = (x_err - x_prev) / dt;
    const float dy = (y_err - y_prev) / dt;

    //Anti-windup
    const float w_z = std::exp(-std::abs(z_err) / coefs[5]);
    const float w_x = std::exp(-std::abs(x_err) / coefs[6]);
    const float w_y = std::exp(-std::abs(y_err) / coefs[7]);

    const float integral_limit_z = coefs[8] / (std::abs(dz) + eps);
    const float integral_limit_x = coefs[9] / (std::abs(dx) + eps);
    const float integral_limit_y = coefs[10] / (std::abs(dy) + eps);

    z_integral += z_err * dt * w_z * coefs[11];
    z_integral = clamp_integral(z_integral, -integral_limit_z, integral_limit_z);

    x_integral += x_err * dt * w_x * coefs[12];
    x_integral = clamp_integral(x_integral, -integral_limit_x, integral_limit_x);

    y_integral += y_err * dt * w_y * coefs[13];
    y_integral = clamp_integral(y_integral, -integral_limit_y, integral_limit_y);


    z_prev = z_err;
    x_prev = x_err;
    y_prev = y_err;


    float altitude_pid = coefs[14] * dyn_gain * z_err
                         + coefs[15] * dyn_gain * dz
                         + z_integral;
    float x_pid = coefs[16] * dyn_gain * x_err
                  + coefs[17] * dyn_gain * dx
                  + x_integral;

    float y_pid = coefs[18] * dyn_gain * y_err
                  + coefs[19] * dyn_gain * dy
                  + y_integral;


    const Eigen::Vector3f target_velocity{x_pid, y_pid, altitude_pid};
    const Eigen::Vector3f current_velocity{vx_smooth, vy_smooth, vz_smooth};

    Eigen::Vector3f target_acceleration = target_velocity - current_velocity + gravity_vec;

    Eigen::Vector3f thrust_direction;
    float norm_acc = target_acceleration.norm();
    if (norm_acc > 1e-6) {
        thrust_direction = target_acceleration / norm_acc;
    } else {
        thrust_direction = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }

    float pitch = std::atan2(thrust_direction[0], thrust_direction[2]);
    float roll  = -std::atan2(thrust_direction[1], thrust_direction[2]);

    constexpr float yaw = 0.0f;
    constexpr float max_tilt = 1.0f;
    roll = std::clamp(roll, -max_tilt, max_tilt);
    pitch = std::clamp(pitch, -max_tilt, max_tilt);


    Eigen::Matrix3f R_desired = euler_to_rotation(roll, pitch, yaw);
    Eigen::Matrix3f R_current = euler_to_rotation(ypr[2], ypr[1], ypr[0]);
    Eigen::Matrix3f R_err = R_current.transpose() * R_desired;

    Eigen::Vector3f ori_err = rotation_matrix_to_axis_angle(R_err);
    ori_err *= 4.0f * coefs[20];

    Eigen::Vector3f desired_omega = ori_err;
    Eigen::Vector3f state_omega{ox_smooth, oy_smooth, oz_smooth};
    Eigen::Vector3f omega_err = desired_omega - state_omega;

    Eigen::Vector3f Kp_ang(coefs[21], coefs[22], coefs[23]);

    Eigen::Vector3f body_torque_des = J * Kp_ang.asDiagonal() * omega_err + state_omega.cross(J * state_omega);

    float required_thrust = mass * target_acceleration[2];
    required_thrust = std::max(required_thrust, 0.0f);

    Eigen::Vector<float,4> output;
    output << required_thrust, body_torque_des.x(), body_torque_des.y(), body_torque_des.z();

    publish_thrust_torque_setpoint(output);
}


void OffboardControl::rlControl()
{
    return;
}

/**
 * @brief Calculate the thrust and attitude setpoint to send to PX4
 */
void OffboardControl::calculate_thrust_torque_setpoint()
{
    // Gather dt in s .. (current_time - prev_time)
    bool debug = false;
    auto now = std::chrono::steady_clock::now();

    if (prev_time.time_since_epoch().count() == 0) {
        prev_time = now;
        return;
    }

    float dt = std::chrono::duration<float>(now - prev_time).count();

    if (dt < 0.0075)
    {
        return;
    }
    prev_time = now;
    if (debug) std::cout << "[DEBUG] dt: " << dt << std::endl;

    // float target_position_x = 2.0f;
    // float target_position_y = local_position_y_;
    // float target_postion_z = 2.5f;


    // print         traj_log_ << "time,x,y,z,type,lap" << std::endl;

    Eigen::Vector3f position = getPositionAtTime(t * 0.004).head(3);
    // Eigen::Vector3f position(0.0f, 0.0f, 4.0f);
    float target_yaw = getPositionAtTime(t * 0.004)[3];
    t++;
    float target_position_x = position[0];
    float target_position_y = position[1];
    float target_postion_z = position[2];

    traj_log_ <<  local_position_x_ << "," << local_position_y_ << "," << local_position_z_ << "," << target_position_x << "," << target_position_y << "," << target_postion_z << "," << motor_0 << "," << motor_1 << "," << motor_2 << "," << motor_3 << "," << target_yaw << "," << local_ori_z_ << std::endl;


    // PID - start
    Eigen::Vector<float,1> altitude_error{target_postion_z - local_position_z_};
    if (debug) std::cout << "[DEBUG] Altitude error: " << altitude_error[0] << std::endl;

    Eigen::Vector<float,1> altitude_control;
    altitude_control[0] = pid_map_["pid_z"].kp * altitude_error[0];
    altitude_control[0] += pid_map_["pid_z"].ki * z_i;
    z_i += altitude_error[0] * dt;
    // fmod to prevent windup
    z_i = fmod(z_i, 0.5);

    if(altitude_prev != 0)
    {
        altitude_control[0] += pid_map_["pid_z"].kd * (altitude_error[0] - altitude_prev) / dt;
    }


    // if(!is_first_run)
    // {
    //     altitude_control[0] += .002 * local_velocity_z_ / dt;
    // }

    if (debug) std::cout << "[DEBUG] Altitude control: " << altitude_control[0] << std::endl;
    altitude_prev = altitude_error[0];

    Eigen::Vector<float,1> x_error(target_position_x - local_position_x_);
    if (debug) std::cout << "[DEBUG] X error: " << x_error[0]  << std::endl;
    Eigen::Vector<float,1> x_control;
    x_control[0] = pid_map_["pid_x"].kp * x_error[0];
    x_control[0] += pid_map_["pid_x"].ki * x_i;
    x_i += x_error[0] * dt;
    // fmod to prevent windup
    x_i = fmod(x_i, 0.5);
    if(x_prev != 0)    // if(!is_first_run)
        // {
        //     // I term for P control (disturbance observer)
        //     Eigen::Vector<float, 3> xdot = Eigen::Vector<float,3>(local_velocity_x_, local_velocity_y_, local_velocity_z_);
        //     // std::cout << "[DEBUG] Local velocity: " << xdot.transpose() << std::endl;

        //     Eigen::Vector<float, 3> xdot_pred = prev_cmd_vel;

        //     // std::cout << "[DEBUG] Predicted velocity: " << xdot_pred.transpose() << std::endl;

        //     Eigen::Vector<float, 3> ddot = L.cwiseProduct(xdot - xdot_pred);
        //     // std::cout << "[DEBUG] ddot: " << ddot.transpose() << std::endl;
        //     // std::cout << "[DEBUG] Estimated disturbance d: " << d.transpose() << std::endl;
        //     d += ddot * dt;

        //     std::cout << "[DEBUG] ddot: " << ddot.transpose() << std::endl;
        //     std::cout << "[DEBUG] Estimated disturbance d: " << d.transpose() << std::endl;
        // }
    {
        x_control[0] += pid_map_["pid_x"].kd * (x_error[0] - x_prev) / dt;
    }
    x_prev = x_error[0];
    if (debug) std::cout << "[DEBUG] X control: " << x_control[0] << std::endl;

    Eigen::Vector<float,1> y_error(target_position_y - local_position_y_);
    if (debug) std::cout << "[DEBUG] Y error: " << y_error[0] << std::endl;

    Eigen::Vector<float,1> y_control;
    y_control[0] = pid_map_["pid_y"].kp * y_error[0];
    y_control[0] += pid_map_["pid_y"].ki * y_i;
    y_i += y_error[0] * dt;
    // fmod to prevent windup
    y_i = fmod(y_i, 0.5);
    if(y_prev != 0)
    {
        y_control[0] += pid_map_["pid_y"].kd * (y_error[0] - y_prev) / dt;
    }
    y_prev = y_error[0];
    if (debug) std::cout << "[DEBUG] Y control: " << y_control[0] << std::endl;

    Eigen::Vector<float,3> target_velocity(x_control[0], y_control[0], altitude_control[0]);
    // target_velocity -= d;
    if (debug) std::cout << "[DEBUG] Target velocity: " << target_velocity.transpose() << std::endl;

    const Eigen::Vector<float, 3> gv3 = {0.0, 0.0, 9.81};
    Eigen::Vector<float,3> target_acceleration = target_velocity - Eigen::Vector<float,3>(local_velocity_x_, local_velocity_y_, local_velocity_z_) + gv3;
    if (debug) std::cout << "[DEBUG] Target acceleration: " << target_acceleration.transpose() << std::endl;

    // {
    //     // I term for P control (disturbance observer)
    //     Eigen::Vector<float, 3> xdot = Eigen::Vector<float,3>(local_velocity_x_, local_velocity_y_, local_velocity_z_);
    //     Eigen::Vector<float, 3> xdot_pred = model(prev_collective_thrust_, dt);
    //     Eigen::Vector<float, 3> ddot = L.cwiseProduct(xdot - xdot_pred);
    //     d += ddot * dt;
    //     // std::cout << "[DEBUG] ddot: " << ddot.transpose() << std::endl;
    //     // std::cout << "[DEBUG] Estimated disturbance d: " << d.transpose() << std::endl;
    // }
    Eigen::Vector<float,3> thrust_direction = target_acceleration.normalized();
    if (debug) std::cout << "[DEBUG] Thrust direction (normalized): " << thrust_direction.transpose() << std::endl;

    float pitch = std::atan2(thrust_direction[0], thrust_direction[2]);
    float roll  = -std::atan2(thrust_direction[1], thrust_direction[2]);

    // float roll = 0.5f;
    // float pitch = 0.0f;
    // std::cout << "Desired roll: " << roll << ", Actual roll: " << local_ori_x_ << std::endl;

    // // limit max pitch/roll to 0.5 rad
    pitch = std::clamp(pitch, -1.0f, 1.0f);
    roll = std::clamp(roll, -1.0f, 1.0f);

    // we will test roll/pitch control

    // float roll = 0.5;
    // float pitch = 0.0;
    float yaw   = 0.0;
    if (debug) std::cout << "[DEBUG] Desired angles - Pitch: " << pitch << ", Roll: " << roll << ", Yaw: " << yaw << std::endl;

    Eigen::Matrix3f R_current = euler_to_rotation(local_ori_x_, local_ori_y_, local_ori_z_);
    Eigen::Matrix3f R_desired = euler_to_rotation(roll, pitch, yaw);
    Eigen::Matrix3f R_err = R_current.transpose() * R_desired;
    Eigen::Vector3f ori_error = rotation_matrix_to_axis_angle(R_err);
    if (debug) std::cout << "[DEBUG] Orientation error (axis-angle): " << ori_error.transpose() << std::endl;
    if (debug)std::cout << "[DEBUG] Roll axis error: " << ori_error.x() << " (should be positive for positive roll)\n";
    // I-term for orientation control
    ori_i += ori_error * dt;
    // fmod to prevent windup
    ori_i = ori_i.cwiseMax(-0.5).cwiseMin(0.5);
    Eigen::Vector3f desired_omega = pid_map_["ori_xyz"].kp * ori_error; // P control
    if(ori_prev[0] != 0)
    {
        desired_omega += pid_map_["ori_xyz"].kd * (ori_error - ori_prev) / dt;
    }
    ori_prev = ori_error;
    // Calculate required thrust and desired angular velocities
    float required_thrust = mass * target_acceleration(2);
    if (debug) std::cout << "[DEBUG] Required thrust: " << required_thrust << std::endl;
    if (debug) std::cout << "[DEBUG] Desired omega: " << desired_omega.transpose() << std::endl;

    Eigen::Vector<float,3> state_omega = Eigen::Vector<float, 3>(local_angular_velocity_x_, local_angular_velocity_y_, local_angular_velocity_z_);
    Eigen::Vector<float,3> omega_err = desired_omega - state_omega;
    if (debug) std::cout << "[DEBUG] Omega error: " << omega_err.transpose() << std::endl;

    Eigen::Vector<float,3> body_torque_des = J * pGainBodyRates * omega_err + state_omega.cross(J * state_omega);
    if (debug) std::cout << "[DEBUG] Body torque desired: " << body_torque_des.transpose() << std::endl;

    Eigen::Vector<float,4> thrust_and_torque(
        required_thrust,
        body_torque_des.x(),
        body_torque_des.y(),
        body_torque_des.z()
    );
    if (debug) std::cout << "[DEBUG] Thrust and torque vector: " << thrust_and_torque.transpose() << std::endl;


    // if(!is_first_run)
    // {
    //     // I term for P control (disturbance observer)
    //     Eigen::Vector<float, 3> xdot = Eigen::Vector<float,3>(local_velocity_x_, local_velocity_y_, local_velocity_z_);
    //     // std::cout << "[DEBUG] Local velocity: " << xdot.transpose() << std::endl;

    //     Eigen::Vector<float, 3> xdot_pred = prev_cmd_vel;

    //     // std::cout << "[DEBUG] Predicted velocity: " << xdot_pred.transpose() << std::endl;

    //     Eigen::Vector<float, 3> ddot = L.cwiseProduct(xdot - xdot_pred);
    //     // std::cout << "[DEBUG] ddot: " << ddot.transpose() << std::endl;
    //     // std::cout << "[DEBUG] Estimated disturbance d: " << d.transpose() << std::endl;
    //     d += ddot * dt;

    //     std::cout << "[DEBUG] ddot: " << ddot.transpose() << std::endl;
    //     std::cout << "[DEBUG] Estimated disturbance d: " << d.transpose() << std::endl;
    // }

    prev_local_velocity_x_ = local_velocity_x_;
    prev_local_velocity_y_ = local_velocity_y_;
    prev_local_velocity_z_ = local_velocity_z_;
    prev_collective_thrust_ = required_thrust;
    prev_ori_q = ori_q;
    prev_cmd_vel = target_velocity;

    is_first_run = false;


    // Publish the thrust and torque setpoint
    publish_thrust_torque_setpoint(thrust_and_torque);
}

Eigen::Vector<float, 3> OffboardControl::model(float prev_collective_thrust_, float dt)
{
    Eigen::Vector<float, 3> lv = Eigen::Vector<float,3>(prev_local_velocity_x_, prev_local_velocity_y_, prev_local_velocity_z_);
    // calculate motor thrusts givenu(t)
    Eigen::Vector<float, 3> force(0, 0, prev_collective_thrust_);
    // just take ori_q from sensors for now

    Eigen::Vector<float, 3> la = (prev_ori_q * force * 1.0 / mass + Eigen::Vector<float, 3>(0, 0, -9.81)); // could be -9.81 or 9.81. I have no idea

    la *= dt;

    return lv + la + d;
}

void OffboardControl::precomputeTrajectory(float dt)
{
    const float total_duration_ = 40.0f;
    float dt_ = 0.01f;

    const float max_radius = 0.9f;
    const float T         = total_duration_ / 2.0f; // one figure-8
    const float amplitude_z = 0.3f;

    traj_.clear();
    const int N = static_cast<int>(std::ceil(total_duration_ / dt_)) + 1;
    traj_.reserve(N);

    // fixed rotation for diagonal lemniscate
    const float theta = static_cast<float>(M_PI) / 4.0f;
    const float cR = std::cos(theta);
    const float sR = std::sin(theta);

    float yaw_prev = std::numeric_limits<float>::infinity();

    for (int i = 0; i < N; ++i) {
        float t = i * dt_;
        float t_mod = std::fmod(t, total_duration_);
        bool second_loop = (t_mod >= T);
        float local_t = second_loop ? t_mod - T : t_mod;

        float angle = 2.0f * static_cast<float>(M_PI) * (local_t / T);

        // base lemniscate
        float x0 = max_radius * std::sin(angle);
        float y0 = max_radius * std::sin(2.0f * angle);

        // rotate to diagonal
        float x = cR * x0 - sR * y0;
        float y = sR * x0 + cR * y0;

        // z
        float z = t_z_ + (second_loop ? -1.0f : 1.0f) * amplitude_z * std::sin(angle);

        // yaw (kept for completeness; can be dropped if unused)
        const float c1 = std::cos(angle);
        const float c2 = std::cos(2.0f * angle);
        float yaw = std::atan2(2.0f * c2, c1);

        if (std::isfinite(yaw_prev)) {
            float dyaw = yaw - yaw_prev;
            while (dyaw >  static_cast<float>(M_PI)) dyaw -= 2.0f * static_cast<float>(M_PI);
            while (dyaw <= -static_cast<float>(M_PI)) dyaw += 2.0f * static_cast<float>(M_PI);
            yaw = yaw_prev + dyaw;
        }
        yaw_prev = yaw;

        traj_.emplace_back(x, y, z, yaw);
    }
}

/**
 * @brief Get the estimated position at time T
 */
Eigen::Vector4f OffboardControl::getPositionAtTime(float t)
{
    if (traj_.empty()) {
        // Fallback: either assert, or compute analytically here.
        return Eigen::Vector4f::Zero();
    }

    // Clamp or wrap time into [0, total_duration_]. NOTE TOTAL_DURATION_ IS HARDCODED 40.0f
    if (t >= 40.0f) {
        finished_ = true;
        t = std::fmod(t, 40.0f);
    } else if (t < 0.0f) {
        t = 0.0f;
    }

    float idx_f = t / 0.01f; // dt_ is 0.01f
    int idx = static_cast<int>(idx_f);

    if (idx >= static_cast<int>(traj_.size()) - 1) {
        // Last point
        return traj_.back();
    }

    float alpha = idx_f - static_cast<float>(idx); // interpolation factor [0,1]
    const Eigen::Vector4f& p0 = traj_[idx];
    const Eigen::Vector4f& p1 = traj_[idx + 1];

    // Linear interpolation
    Eigen::Vector4f p = (1.0f - alpha) * p0 + alpha * p1;
    return p;
}

/**
 * @brief Command the vehicle to the starting position.
 *       Currently, this only confirms target altitiude and not position.
 */
void OffboardControl::get_to_start_pos()
{
    // Lift off phase to reach target altitude
    // d_thrust_ = 0.6 * 9.81; // Assuming measured in N
    // std::cout << "Not ascended yet" << std::endl;
    px4_msgs::msg::TrajectorySetpoint setpoint;
    setpoint.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    setpoint.position = {0, 0, -t_z_};
    setpoint.velocity = {NAN, NAN, NAN};
    setpoint.acceleration = {NAN, NAN, NAN};
    setpoint.jerk = {NAN, NAN, NAN};
    setpoint.yaw = NAN;
    setpoint.yawspeed = NAN;
    trajectory_setpoint_publisher_->publish(setpoint);

    if (std::fabs(local_position_x_ - 0) < 0.3f &&
            std::fabs(local_position_y_ - 0) < 0.3f &&
            std::fabs(std::fabs(local_position_z_) - t_z_) < 0.3f
       ) // NED -> ENU
    {
        std::cout << "Hit Acended counter: " << ascended_counter_ << std::endl;
        ascended_counter_++;
        // actuator_enabled_ = false;
    }
    if (ascended_counter_ > 20)
    {
        ascended_ = true;
        start_time_ = std::chrono::steady_clock::now();
        std::cout << "Ascended" << std::endl;
    }
}

// /**
//  * @brief If the quad has ascended, check current position against waypoints and update trajectory
//  */
// void OffboardControl::update_trajectory()
// {
//     // Trajectory tracking
//     // Get the current waypoint

//     if(laps > 2)
//     {
//         // Disarm the vehicle
//         finished_ = true;
//         float time = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time_).count();
//         std::cout << "Finished in " << time << " seconds" << std::endl;
//     }
//     float x = waypoints_[current_waypoint_][0];
//     float y = waypoints_[current_waypoint_][1];
//     float z = waypoints_[current_waypoint_][2];

//     if (std::fabs(local_position_x_ - x) < 0.1f &&
//             std::fabs(local_position_y_ - y) < 0.1f &&
//             std::fabs(local_position_z_ - z) < 0.1f
//        ) // NED -> ENU
//     {
//         std::cout << "new waypoint" << std::endl;
//         std::cout << "xyz: " << local_position_x_ << ", " << local_position_y_ << ", " << local_position_z_ << std::endl;
//         std::cout << "waypoint: " << x << ", " << y << ", " << z << std::endl;

//         current_waypoint_++;
//         if (current_waypoint_ == 64)
//         {
//             current_waypoint_ = 0;
//             laps++;
//         }
//     }
// }


/**
 * @brief Manage the flight mode of the drone based on current state flags
 * !ascended && !finished_ -> get_to_start_pos()
 * ascended && !finished_ -> update_trajectory()
 * finished_ && ascended_ -> land()
 */
void OffboardControl::state_manager()
{
    if (!ascended_ && !finished_)
    {
        get_to_start_pos();
    }
    else if (ascended_ && !finished_)
    {
        // rvolmeaControl();
        lqrControl();
    }
    else if (finished_ && ascended_)
    {
        // Land the vehicle,
        if(!told_to_land)
        {
            land();
            told_to_land = true;
        }
    }
}
/**
 * @brief Publish the offboard control mode.
 */
void OffboardControl::publish_offboard_control_mode()
{
    OffboardControlMode msg{};
    msg.position = !ascended_ && !finished_;
    msg.velocity = false;
    msg.acceleration = false;
    msg.attitude = false;
    msg.body_rate = ascended_ && !finished_;
    msg.thrust_and_torque = ascended_ && !finished_;
    msg.direct_actuator = false;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    offboard_control_mode_publisher_->publish(msg);
}


/**
 * @brief Publish vehicle commands
 * @param command   Command code (matches VehicleCommand and MAVLink MAV_CMD codes)
 * @param param1    Command parameter 1
 * @param param2    Command parameter 2
 */
void OffboardControl::publish_vehicle_command(uint16_t command, float param1, float param2)
{
    VehicleCommand msg{};
    msg.param1 = param1;
    msg.param2 = param2;
    msg.command = command;
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    vehicle_command_publisher_->publish(msg);
}

void OffboardControl::lqrControl()
{
    auto now = std::chrono::steady_clock::now();
    if (prev_time.time_since_epoch().count() == 0) {
        prev_time = now;
        return;
    }
    float dt = std::chrono::duration<float>(now - prev_time).count();
    prev_time = now;

    // 1. Get State
    Eigen::Vector3f position(local_position_x_, local_position_y_, local_position_z_);
    Eigen::Vector3f velocity(local_velocity_x_, local_velocity_y_, local_velocity_z_);
    Eigen::Vector3f euler(local_ori_x_, local_ori_y_, local_ori_z_); // Roll, Pitch, Yaw
    Eigen::Vector3f body_rates(local_angular_velocity_x_, local_angular_velocity_y_, local_angular_velocity_z_);

    // 2. Get Target
    Eigen::Vector4f target = getPositionAtTime(t * 0.004f); // Same time scaling as rvolmea
    t++;
    Eigen::Vector3f target_position = target.head(3);
    // Eigen::Vector3f target_position = Eigen::Vector3f(0.0f, 0.0f, 4.0f);
    float target_yaw = 0.0f;

    // Target velocity/acceleration/rates assumed 0 for simple waypoint tracking or slow trajectory
    Eigen::Vector3f target_velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f target_euler = Eigen::Vector3f(0, 0, target_yaw);
    Eigen::Vector3f target_body_rates = Eigen::Vector3f::Zero();

    // 3. Calculate Error State vector (12x1)
    // x = [pos_err, vel_err, ori_err, rate_err]
    Eigen::Vector<float, 12> state_error;

    state_error.segment<3>(0) = position - target_position;
    state_error.segment<3>(3) = velocity - target_velocity;

    // Orientation error handling (simple difference for small angles, but handle yaw wrap)
    Eigen::Vector3f ori_err = euler - target_euler;
    if (ori_err[2] > M_PI) ori_err[2] -= 2 * M_PI;
    if (ori_err[2] < -M_PI) ori_err[2] += 2 * M_PI;
    state_error.segment<3>(6) = ori_err;

    state_error.segment<3>(9) = body_rates - target_body_rates;

    // 4. LQR Gains (K) - 4x12 Matrix
    // u = [Thrust, TorqueX, TorqueY, TorqueZ]
    // u = u_eq - K * x_err
    // Note: This assumes the system is linearized around hover.
    // u_eq = [mg, 0, 0, 0]

    if (!k_lqr_initialized) {
        RCLCPP_WARN(this->get_logger(), "LQR Gains not initialized! Computing now (this should have happened at startup).");
        computeLQRGains();
    }

    // 5. Calculate Control Input
    Eigen::Vector4f u_eq(mass * 9.81f, 0.0f, 0.0f, 0.0f);
    Eigen::Vector4f u = u_eq - K_lqr * state_error;
    // std::cout << "[LQR DEBUG] State Error: " << state_error.transpose() << std::endl;
    // std::cout << "[LQR DEBUG] Control Input (u): " << u.transpose() << std::endl;
    // std::cout << "[LQR DEBUG] u_eq: " << u_eq.transpose() << std::endl;
    // std::cout << "[LQR DEBUG] K * error: " << (K_lqr * state_error).transpose() << std::endl;

    // 6. Publish
    publish_thrust_torque_setpoint(u);

    // Logging
    traj_log_ << local_position_x_ << "," << local_position_y_ << "," << local_position_z_ << ","
              << target_position[0] << "," << target_position[1] << "," << target_position[2] << ","
              << motor_0 << "," << motor_1 << "," << motor_2 << "," << motor_3 << ","
              << target_yaw << "," << local_ori_z_ << std::endl;
}

void OffboardControl::computeLQRGains()
{
    // Define System Matrices Linearized around Hover
    // Frame: ENU (East-North-Up) / FLU (Forward-Left-Up)
    // State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    // Input: [Thrust, TorqueX, TorqueY, TorqueZ] (normalized or physical)

    float g = 9.81f;

    // A Matrix (12x12)
    Eigen::Matrix<float, 12, 12> A = Eigen::Matrix<float, 12, 12>::Zero();

    // Position derivatives are velocities
    A(0, 3) = 1.0f;
    A(1, 4) = 1.0f;
    A(2, 5) = 1.0f;

    // Velocity derivatives (linearized gravity/tilt)
    // x_dot_dot = g * theta
    A(3, 7) = g;
    // y_dot_dot = -g * phi
    A(4, 6) = -g;
    // z_dot_dot = 0 (controlled by thrust directly)

    // Orientation derivatives are rates
    A(6, 9) = 1.0f;
    A(7, 10) = 1.0f;
    A(8, 11) = 1.0f;

    // B Matrix (12x4)
    Eigen::Matrix<float, 12, 4> B = Eigen::Matrix<float, 12, 4>::Zero();

    // Vertical acceleration from Thrust (u1)
    // z_dot_dot = 1/m * u1 (Positive thrust = Upward acceleration in ENU)
    B(5, 0) = 1.0f / mass;

    // Angular accelerations from Torques
    // p_dot = 1/Ix * u2
    // q_dot = 1/Iy * u3
    // r_dot = 1/Iz * u4
    // Assuming diagonal inertia J
    B(9, 1) = 1.0f / J(0,0);
    B(10, 2) = 1.0f / J(1,1);
    B(11, 3) = 1.0f / J(2,2);

    // Q Matrix (State Cost)
    Eigen::Matrix<float, 12, 12> Q = Eigen::Matrix<float, 12, 12>::Identity();
    Q.diagonal() << lqr_params.Q_pos_xy, lqr_params.Q_pos_xy, lqr_params.Q_pos_z,   // Pos (x,y,z)
               lqr_params.Q_vel_xy, lqr_params.Q_vel_xy, lqr_params.Q_vel_z,   // Vel
               lqr_params.Q_ang_xy, lqr_params.Q_ang_xy, lqr_params.Q_ang_z,   // Angle (phi, theta, psi)
               lqr_params.Q_rate_xy, lqr_params.Q_rate_xy, lqr_params.Q_rate_z; // Rates

    // R Matrix (Input Cost)
    Eigen::Matrix<float, 4, 4> R = Eigen::Matrix<float, 4, 4>::Identity();
    R.diagonal() << lqr_params.R_thrust, lqr_params.R_torque_xy, lqr_params.R_torque_xy, lqr_params.R_torque_z;

    // Solve Algebraic Riccati Equation (Continuous)
    // A^T P + P A - P B R^-1 B^T P + Q = 0
    // We can use a simple iterative solver here since we don't have a dedicated library function exposed easily

    Eigen::Matrix<float, 12, 12> P = Q; // Initial guess
    Eigen::Matrix<float, 12, 12> P_next;
    float dt_sim = 0.001f; // Simulation step for discrete approximation or gradient descent

    // Simple Gradient Descent / Integration to steady state
    // dP/dt = A^T P + P A - P B R^-1 B^T P + Q
    // We iterate until dP/dt is close to 0

    Eigen::Matrix<float, 4, 4> R_inv = R.inverse();

    for(int i=0; i<10000; i++) {
        Eigen::Matrix<float, 12, 12> dP = A.transpose() * P + P * A - P * B * R_inv * B.transpose() * P + Q;
        P += dP * dt_sim;

        if(dP.norm() < 1e-5) {
            // std::cout << "LQR Converged in " << i << " iterations." << std::endl;
            break;
        }
    }

    // K = R^-1 B^T P
    K_lqr = R_inv * B.transpose() * P;

    k_lqr_initialized = true;

    std::cout << "Calculated LQR Gain K:\n" << K_lqr << std::endl;
}

int main(int argc, char *argv[])
{
    std::cout << "Starting offboard control node..." << std::endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OffboardControl>());

    rclcpp::shutdown();
    return 0;
}
