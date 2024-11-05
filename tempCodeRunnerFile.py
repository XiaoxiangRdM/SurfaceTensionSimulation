                centroid = solid_obj.centroid
                ax.scatter(centroid[0], centroid[1], centroid[2], color='r', s=50, label="Centroid" if i == 0 else None)

                # Draw each face
                for face in faces:
                    x = vertices[face, 0]
                    y = vertices[face, 1]
                    z = vertices[face, 2]
                    ax.plot_trisurf(x, y, z, color=(0.7, 0.7, 0.7, 0.5), edgecolor='k', linewidth=0.2)

                # Orientation axes
                rotation_matrix = solid_obj.rotation_matrix
                # Draw orientation axes for each solid
                for j in range(3):
                    axis = rotation_matrix[:, j]
                    ax.quiver(centroid[0], centroid[1], centroid[2],
                              axis[0], axis[1], axis[2],
                              color=['r', 'g', 'b'][j], length=1.0)

            plt.pause(0.01)
