### Pull Request Workflow

1. **Create a Feature Branch:**
   - Create a new branch from `main` to work on your feature or fix.
     ```bash
     git checkout -b feature-branch-name main
     ```

2. **Make and Commit Changes:**
   - Make your changes locally and commit them.
     ```bash
     git add .
     git commit -m "Your descriptive commit message"
     ```

3. **Push the Feature Branch:**
   - Push your feature branch to the remote repository.
     ```bash
     git push origin feature-branch-name
     ```

4. **Create a Pull Request (PR):**
   - Navigate to your repository on GitHub.
   - Click on "Compare & pull request" for the feature branch.
   - Fill out the PR details, describing your changes and mentioning relevant issues.

5. **Review and Discuss:**
   - Team members review your code on GitHub, provide feedback, and suggest changes.

6. **Address Feedback:**
   - Make additional commits to address feedback on your feature branch.

7. **Merge the Pull Request:**
   - Once approved and all feedback is addressed, merge the PR into `main`.
   - Update your local `main` branch if necessary (`git pull origin main`).
   - Merge the PR on GitHub using the "Merge pull request" button.

8. **Delete the Feature Branch (optional):**
   - After merging, delete the feature branch on GitHub.

9. **Celebrate 🎉**

