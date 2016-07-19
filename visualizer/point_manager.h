#ifndef _POINT_MANAGER_H
#define _POINT_MANAGER_H

#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <GL/gl.h>

class Renderable;

typedef std::function<bool (GLfloat *, GLfloat *, int)> fill_data_func;

class PointManager {
  public:
    PointManager(Renderable *point_renderable, int n_points_,
                 fill_data_func fill_func_);
    ~PointManager();
    void start_update_thread();
    void maybe_update();

  private:
    void update_thread();

    Renderable *renderable;
    int n_points;
    fill_data_func fill_func;

    bool new_data;
    int ready_ind;
    std::vector<GLfloat> ptr_pos[2];
    std::vector<GLfloat> ptr_color[2];

    std::mutex mtx;
    std::thread thread;
    std::atomic_bool exit;
};

#endif
