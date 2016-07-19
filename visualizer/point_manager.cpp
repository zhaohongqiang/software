#include "point_manager.h"

#include "graphics_engine/renderable.h"

PointManager::PointManager(Renderable *point_renderable, int n_points_,
                           fill_data_func fill_func_)
  : renderable(point_renderable),
    n_points(n_points_),
    fill_func(fill_func_), new_data(false), ready_ind(0), exit(false) {
  ptr_pos[0].reserve(n_points * 3);
  ptr_pos[1].reserve(n_points * 3);
  ptr_color[0].reserve(n_points * 3);
  ptr_color[1].reserve(n_points * 3);
}

PointManager::~PointManager() {
  exit.store(true);
  thread.join();
}

void PointManager::maybe_update() {
  std::lock_guard<std::mutex> lck(mtx);

  if (new_data) {
    renderable->update_attribute("in_position", ptr_pos[ready_ind].data());
    renderable->update_attribute("in_color", ptr_color[ready_ind].data());
    new_data = false;
  }
}

void PointManager::start_update_thread() {
  thread = std::thread(&PointManager::update_thread, this);
}

void PointManager::update_thread() {
  while (!exit.load()) {
    if (fill_func(ptr_pos[1 - ready_ind].data(),
              ptr_color[1 - ready_ind].data(),
              n_points)) {

      {
        std::lock_guard<std::mutex> lck(mtx);
        new_data = true;
        ready_ind = 1 - ready_ind;
      }
    }
  }
}
